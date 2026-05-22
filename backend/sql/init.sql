-- Protein Subcellular Localization — Database Initialization
-- Run this before starting the Spring Boot application

CREATE DATABASE IF NOT EXISTS protein_localization
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE protein_localization;

CREATE TABLE IF NOT EXISTS sequences (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sequence_id VARCHAR(64) NOT NULL UNIQUE,
    raw_sequence TEXT NOT NULL,
    sequence_length INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sequence_id (sequence_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sequence_id VARCHAR(64) NOT NULL,
    predicted_location VARCHAR(64) NOT NULL,
    location_confidence DOUBLE NOT NULL,
    predicted_membrane VARCHAR(32),
    membrane_confidence DOUBLE,
    all_probabilities JSON,
    attention_data JSON,
    model_version VARCHAR(32) DEFAULT 'v1',
    inference_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sequence_id (sequence_id),
    INDEX idx_created_at (created_at),
    CONSTRAINT fk_predictions_sequence
        FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
