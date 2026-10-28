-- =====================================================================
-- Base de modelagem: uma linha por pedido entregue que recebeu avaliação
-- =====================================================================
--
-- Momento da previsão (T): data de envio da pesquisa de satisfação
-- (review_creation_date). A Olist envia a pesquisa no dia seguinte à
-- entrega ou, se o pedido ainda não chegou, logo após o prazo prometido.
--
-- Regra de ouro contra vazamento de dados: toda variável abaixo usa apenas
-- informações com data ANTERIOR a T. O texto da avaliação, a data da
-- resposta e a própria nota só aparecem no alvo.
--
-- As tabelas brutas (orders, order_items, order_reviews, ...) são
-- registradas como views pelo código Python antes desta consulta rodar.
-- =====================================================================

-- 1. Uma avaliação por pedido: a primeira pesquisa enviada.
--    (543 pedidos têm duas avaliações; ficamos com a primeira.)
WITH avaliacao AS (
    SELECT
        order_id,
        review_score,
        review_creation_date    AS data_pesquisa,
        review_answer_timestamp AS data_resposta
    FROM order_reviews
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY order_id
        ORDER BY review_creation_date, review_answer_timestamp
    ) = 1
),

-- 2. Itens agregados por pedido.
itens AS (
    SELECT
        i.order_id,
        COUNT(*)                                AS n_itens,
        COUNT(DISTINCT i.product_id)            AS n_produtos,
        COUNT(DISTINCT i.seller_id)             AS n_vendedores,
        SUM(i.price)                            AS valor_produtos,
        SUM(i.freight_value)                    AS valor_frete,
        SUM(p.product_weight_g)                 AS peso_total_g,
        SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm)
                                                AS volume_total_cm3,
        AVG(p.product_photos_qty)               AS media_fotos,
        AVG(p.product_description_lenght)       AS media_tamanho_descricao
    FROM order_items i
    LEFT JOIN products p USING (product_id)
    GROUP BY i.order_id
),

-- 3. Item principal (o de maior preço): define vendedor e categoria
--    "principais" do pedido.
item_principal AS (
    SELECT
        i.order_id,
        i.seller_id                                         AS vendedor_principal,
        COALESCE(p.product_category_name, 'sem_categoria')  AS categoria
    FROM order_items i
    LEFT JOIN products p USING (product_id)
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY i.order_id
        ORDER BY i.price DESC, i.order_item_id
    ) = 1
),

-- 4. Pagamentos agregados por pedido.
pagamentos AS (
    SELECT
        order_id,
        ARG_MAX(payment_type, payment_value)        AS tipo_pagamento,
        MAX(payment_installments)                   AS parcelas,
        COUNT(*)                                    AS n_pagamentos,
        MAX((payment_type = 'voucher')::INT)        AS usou_voucher
    FROM order_payments
    GROUP BY order_id
),

-- 5. Coordenada média de cada prefixo de CEP (filtrando pontos fora do
--    Brasil, que existem na tabela de geolocalização).
geo AS (
    SELECT
        geolocation_zip_code_prefix AS cep_prefixo,
        AVG(geolocation_lat)        AS lat,
        AVG(geolocation_lng)        AS lng
    FROM geolocation
    WHERE geolocation_lat BETWEEN -34 AND 6
      AND geolocation_lng BETWEEN -74 AND -34
    GROUP BY 1
),

-- 6. Pedidos entregues com avaliação: o "esqueleto" da base.
pedidos AS (
    SELECT
        o.order_id,
        o.order_purchase_timestamp          AS data_compra,
        o.order_approved_at                 AS data_aprovacao,
        o.order_delivered_carrier_date      AS data_postagem,
        o.order_delivered_customer_date     AS data_entrega,
        o.order_estimated_delivery_date     AS data_prometida,
        a.data_pesquisa,
        a.data_resposta,
        a.review_score,
        c.customer_unique_id,
        c.customer_state                    AS uf_cliente,
        c.customer_zip_code_prefix          AS cep_cliente,
        ip.vendedor_principal,
        ip.categoria,
        s.seller_state                      AS uf_vendedor,
        s.seller_zip_code_prefix            AS cep_vendedor
    FROM orders o
    JOIN avaliacao a        USING (order_id)
    JOIN customers c        USING (customer_id)
    JOIN item_principal ip  USING (order_id)
    LEFT JOIN sellers s     ON s.seller_id = ip.vendedor_principal
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
),

-- 7. Histórico do vendedor: avaliações de pedidos ANTERIORES do mesmo
--    vendedor cuja resposta chegou antes de T. É aqui que mais projetos
--    vazam informação; a condição h.data_resposta < p.data_pesquisa
--    garante que só usamos o que já era conhecido.
historico_base AS (
    SELECT vendedor_principal, data_resposta, (review_score <= 2)::INT AS negativa
    FROM pedidos
),
historico_vendedor AS (
    SELECT
        p.order_id,
        COUNT(h.negativa)           AS vendedor_n_avaliacoes_previas,
        -- COALESCE: vendedor sem histórico tem SUM nulo; vira 0.
        COALESCE(SUM(h.negativa), 0) AS vendedor_n_negativas_previas
    FROM pedidos p
    LEFT JOIN historico_base h
        ON  h.vendedor_principal = p.vendedor_principal
        AND h.data_resposta < p.data_pesquisa
    GROUP BY p.order_id
),

-- 8. Taxa negativa global acumulada até T (usada para suavizar a taxa do
--    vendedor). ASOF JOIN pega a última linha acumulada antes de T.
global_acumulado AS (
    SELECT
        data_resposta,
        COUNT(*)      OVER (ORDER BY data_resposta ROWS UNBOUNDED PRECEDING) AS n_global,
        SUM(negativa) OVER (ORDER BY data_resposta ROWS UNBOUNDED PRECEDING) AS neg_global
    FROM historico_base
),
taxa_global AS (
    SELECT p.order_id, g.n_global, g.neg_global
    FROM pedidos p
    ASOF LEFT JOIN global_acumulado g
        ON g.data_resposta < p.data_pesquisa
)

-- 9. Base final com as variáveis.
SELECT
    p.order_id,
    p.data_compra,
    p.data_pesquisa,
    p.vendedor_principal,

    -- Alvo: nota 1 ou 2.
    (p.review_score <= 2)::INT                                  AS avaliacao_negativa,

    -- Prazo e logística (tudo observado até T).
    DATE_DIFF('hour', p.data_compra, p.data_prometida) / 24.0   AS prazo_prometido_dias,
    DATE_DIFF('minute', p.data_compra, p.data_aprovacao) / 60.0 AS horas_ate_aprovacao,
    -- Postagem anterior à compra é erro de registro (99 pedidos): vira nulo.
    CASE WHEN p.data_postagem < p.data_pesquisa
              AND p.data_postagem >= p.data_compra
         THEN DATE_DIFF('hour', p.data_compra, p.data_postagem) / 24.0
    END                                                         AS dias_ate_postagem,
    (p.data_entrega < p.data_pesquisa)::INT                     AS entregue_antes_pesquisa,
    -- Atraso observado até T: se entregue, atraso real (negativo = adiantado);
    -- se ainda não entregue, quantos dias já passou do prazo.
    CASE WHEN p.data_entrega < p.data_pesquisa
         THEN DATE_DIFF('hour', p.data_prometida, p.data_entrega) / 24.0
         ELSE DATE_DIFF('hour', p.data_prometida, p.data_pesquisa) / 24.0
    END                                                         AS dias_atraso_observado,
    CASE WHEN p.data_entrega < p.data_pesquisa
         THEN DATE_DIFF('hour', p.data_compra, p.data_entrega) / 24.0
    END                                                         AS dias_ate_entrega,

    -- Pedido.
    it.n_itens,
    it.n_produtos,
    it.n_vendedores,
    it.valor_produtos,
    it.valor_frete,
    it.valor_frete / NULLIF(it.valor_produtos, 0)               AS razao_frete_produtos,
    it.peso_total_g,
    it.volume_total_cm3,
    it.media_fotos,
    it.media_tamanho_descricao,
    p.categoria,

    -- Pagamento.
    pg.tipo_pagamento,
    pg.parcelas,
    pg.n_pagamentos,
    pg.usou_voucher,

    -- Geografia.
    p.uf_cliente,
    p.uf_vendedor,
    (p.uf_cliente = p.uf_vendedor)::INT                         AS mesmo_estado,
    -- Distância em linha reta (fórmula de Haversine), em km.
    2 * 6371 * ASIN(SQRT(
        POWER(SIN(RADIANS(gv.lat - gc.lat) / 2), 2)
        + COS(RADIANS(gc.lat)) * COS(RADIANS(gv.lat))
          * POWER(SIN(RADIANS(gv.lng - gc.lng) / 2), 2)
    ))                                                          AS distancia_km,

    -- Calendário da compra.
    DAYOFWEEK(p.data_compra)                                    AS dia_semana_compra,
    HOUR(p.data_compra)                                         AS hora_compra,

    -- Histórico do vendedor até T, com suavização bayesiana:
    -- (negativas + m * taxa_global) / (avaliações + m), com m = 20.
    -- Vendedor novo fica perto da média; vendedor com histórico longo
    -- fica perto da própria taxa. Enquanto a plataforma inteira tiver
    -- menos de 500 avaliações respondidas (início de 2016), a taxa global
    -- não é confiável e a variável fica nula.
    hv.vendedor_n_avaliacoes_previas,
    CASE WHEN tg.n_global >= 500 THEN
        (hv.vendedor_n_negativas_previas + 20 * tg.neg_global / tg.n_global)
          / (hv.vendedor_n_avaliacoes_previas + 20)
    END                                                         AS vendedor_taxa_negativa_previa

FROM pedidos p
LEFT JOIN itens it              USING (order_id)
LEFT JOIN pagamentos pg         USING (order_id)
LEFT JOIN historico_vendedor hv USING (order_id)
LEFT JOIN taxa_global tg        USING (order_id)
LEFT JOIN geo gc                ON gc.cep_prefixo = p.cep_cliente
LEFT JOIN geo gv                ON gv.cep_prefixo = p.cep_vendedor
ORDER BY p.data_pesquisa, p.order_id
