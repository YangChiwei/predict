
#MariaDB 初始化 SQL 指令 (建立 database: taiwanstock 與資料表)
#-- 1. 建立資料庫 taiwanstock
#CREATE DATABASE IF NOT EXISTS taiwanstock CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
#-- 2. 建立專用帳號並授權遠端存取 ('%' 支援本機、區網 Synology NAS 與外網)
#CREATE USER IF NOT EXISTS 'stock_user'@'%' IDENTIFIED BY '12345678';
#GRANT ALL PRIVILEGES ON taiwanstock.* TO 'stock_user'@'%';
#FLUSH PRIVILEGES;

USE taiwanstock;

CREATE TABLE IF NOT EXISTS stock_raw_data (
    stock_id VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    sox_close DOUBLE,
    usd_twd DOUBLE,
    vix_close DOUBLE,
    PRIMARY KEY (stock_id, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE IF NOT EXISTS stock_features (
    stock_id VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    inst_net_buy_20d DOUBLE,
    inst_accel DOUBLE,
    margin_mom DOUBLE,
    revenue_mom DOUBLE,
    ma20_slope DOUBLE,
    ma60_slope DOUBLE,
    bias_ma20 DOUBLE,
    bias_ma60 DOUBLE,
    vol_ratio_5_20 DOUBLE,
    volatility_20d DOUBLE,
    rsi_diff_3d DOUBLE,
    rsi_monthly DOUBLE,
    macd_monthly DOUBLE,
    sox_mom DOUBLE,
    rs_vs_sox DOUBLE,
    usd_twd_mom20 DOUBLE,
    vix_ma20_bias DOUBLE,
    label INT NULL,
    PRIMARY KEY (stock_id, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE IF NOT EXISTS stock_predictions (
    stock_id VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    model_type VARCHAR(20) NOT NULL,
    probability DOUBLE NOT NULL,
    threshold DOUBLE NOT NULL,
    signal_action INT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, date, model_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
