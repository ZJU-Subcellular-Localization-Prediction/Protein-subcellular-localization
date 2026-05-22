package com.proteinlocal.exception;

/**
 * Thrown when Python inference fails with a specific HTTP-equivalent status code.
 * Handlers map the status code directly to the HTTP response.
 */
public class PredictException extends RuntimeException {

    private final int statusCode;

    public PredictException(int statusCode, String message) {
        super(message);
        this.statusCode = statusCode;
    }

    public PredictException(int statusCode, String message, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
    }

    public int getStatusCode() {
        return statusCode;
    }
}
