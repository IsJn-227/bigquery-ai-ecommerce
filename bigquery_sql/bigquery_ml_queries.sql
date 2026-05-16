-- ============================================
-- BigQuery ML: Demand Forecasting
-- Module 1
-- ============================================
CREATE OR REPLACE MODEL `ecommerce.demand_forecast_model`
OPTIONS(
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'transaction_date',
  time_series_data_col = 'total_quantity',
  time_series_id_col = 'product_id',
  horizon = 30,
  auto_arima = TRUE,
  data_frequency = 'DAILY'
) AS
SELECT
  DATE(transaction_date) AS transaction_date,
  product_id,
  SUM(quantity) AS total_quantity
FROM `ecommerce.transactions`
GROUP BY 1, 2;

-- Forecast next 30 days
SELECT
  *
FROM ML.FORECAST(MODEL `ecommerce.demand_forecast_model`,
  STRUCT(30 AS horizon, 0.9 AS confidence_level));


-- ============================================
-- BigQuery ML: Customer Churn Prediction
-- Module 2
-- ============================================
CREATE OR REPLACE MODEL `ecommerce.churn_model`
OPTIONS(
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['is_churned'],
  l2_reg = 0.01
) AS
SELECT
  age, total_spend, num_orders, avg_rating,
  days_since_last_order, segment, is_churned
FROM `ecommerce.customers`;

-- Predict churn probability
SELECT
  customer_id,
  predicted_is_churned,
  predicted_is_churned_probs[OFFSET(1)].prob AS churn_probability
FROM ML.PREDICT(MODEL `ecommerce.churn_model`,
  (SELECT * FROM `ecommerce.customers`));


-- ============================================
-- BigQuery ML: Dynamic Pricing
-- Module 3
-- ============================================
CREATE OR REPLACE MODEL `ecommerce.pricing_model`
OPTIONS(
  model_type = 'LINEAR_REG',
  input_label_cols = ['optimal_price'],
  l2_reg = 1.0
) AS
SELECT
  p.product_id,
  SUM(t.quantity) / COUNT(*) AS demand_score,
  AVG(t.discount) AS avg_discount,
  p.stock_quantity,
  p.rating,
  p.base_price AS optimal_price
FROM `ecommerce.products` p
JOIN `ecommerce.transactions` t USING (product_id)
GROUP BY p.product_id, p.stock_quantity, p.rating, p.base_price;


-- ============================================
-- BigQuery ML: Customer Segmentation (K-Means)
-- Module 6
-- ============================================
CREATE OR REPLACE MODEL `ecommerce.customer_segments`
OPTIONS(
  model_type = 'KMEANS',
  num_clusters = 4,
  standardize_features = TRUE
) AS
SELECT
  total_spend, num_orders, avg_rating,
  days_since_last_order, age
FROM `ecommerce.customers`;

-- Assign segments
SELECT
  customer_id,
  CENTROID_ID AS cluster_id
FROM ML.PREDICT(MODEL `ecommerce.customer_segments`,
  (SELECT * FROM `ecommerce.customers`));


-- ============================================
-- BigQuery ML: Sentiment Analysis
-- Module 10
-- ============================================
CREATE OR REPLACE MODEL `ecommerce.sentiment_model`
OPTIONS(
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['sentiment']
) AS
SELECT
  review_text,
  sentiment
FROM `ecommerce.reviews`
WHERE review_text IS NOT NULL;
