package com.proteinlocal.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.proteinlocal.dto.PredictResponse;
import com.proteinlocal.exception.PredictException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
public class PredictService {

    private static final Logger log = LoggerFactory.getLogger(PredictService.class);

    private final ObjectMapper objectMapper;

    @Value("${predict.python.path:python}")
    private String pythonPath;

    @Value("${predict.python.script:python/predict.py}")
    private String scriptPath;

    @Value("${predict.python.timeout-seconds:60}")
    private int timeoutSeconds;

    @Value("${predict.model.path:best_model.pt}")
    private String modelPath;

    public PredictService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    /**
     * Generate a short hash ID from the raw sequence (MD5 first 12 chars).
     */
    public static String generateSequenceId(String sequence) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(sequence.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : digest) {
                sb.append(String.format("%02x", b));
            }
            return sb.substring(0, 12);
        } catch (Exception e) {
            throw new RuntimeException("MD5 not available", e);
        }
    }

    /**
     * Clean the input sequence: uppercase, strip whitespace, keep only 20 standard amino acids.
     */
    public static String cleanSequence(String raw) {
        if (raw == null) return "";
        return raw.toUpperCase()
                  .replaceAll("\\s+", "")
                  .replaceAll("[^ACDEFGHIKLMNPQRSTVWY]", "");
    }

    /**
     * Run Python inference script and return parsed PredictResponse.
     */
    public PredictResponse predict(String rawSequence) {
        String cleaned = cleanSequence(rawSequence);
        if (cleaned.isEmpty()) {
            throw new IllegalArgumentException("Sequence is empty or contains no valid amino acids");
        }

        String sequenceId = generateSequenceId(cleaned);

        try {
            String json = runPythonInference(cleaned);
            PredictResponse resp = objectMapper.readValue(json, PredictResponse.class);
            resp.setSequenceId(sequenceId);
            return resp;
        } catch (PredictException e) {
            throw e; // re-throw as-is — already has correct status code
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            log.error("Failed to parse Python JSON output for sequence {}", sequenceId, e);
            throw new PredictException(502, "Failed to parse inference result", e);
        } catch (Exception e) {
            log.error("Unexpected error during inference for sequence {}", sequenceId, e);
            throw new PredictException(500, "Inference failed: " + e.getMessage(), e);
        }
    }

    private String runPythonInference(String sequence) throws Exception {
        File scriptFile = new File(scriptPath);
        if (!scriptFile.exists()) {
            scriptFile = new File("../" + scriptPath);
        }
        if (!scriptFile.exists()) {
            throw new PredictException(503,
                    "Python inference script not found: " + scriptPath
                    + ". Please ensure predict.python.script points to a valid file.");
        }
        String absScript = scriptFile.getAbsolutePath();

        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            // model path is relative to script directory
            modelFile = new File(scriptFile.getParentFile(), modelPath);
        }
        if (!modelFile.exists()) {
            log.error("Model checkpoint not found: {}", modelPath);
        }

        ProcessBuilder pb = new ProcessBuilder(
                pythonPath, absScript, "--sequence", sequence, "--model-path", modelPath
        );
        pb.directory(scriptFile.getParentFile());
        pb.redirectErrorStream(false);
        pb.environment().put("HF_HUB_OFFLINE", "1");
        pb.environment().put("TRANSFORMERS_OFFLINE", "1");

        log.info("Running: {} {} --sequence <SEQ>", pythonPath, absScript);

        Process process;
        try {
            process = pb.start();
        } catch (java.io.IOException e) {
            throw new PredictException(503,
                    "Failed to start Python process. Check that "
                    + pythonPath + " is a valid Python executable.", e);
        }

        // Read stdout
        String stdout;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            stdout = reader.lines().collect(Collectors.joining("\n"));
        }

        // Read stderr
        String stderr;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
            stderr = reader.lines().collect(Collectors.joining("\n"));
        }

        boolean finished = process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            throw new PredictException(504,
                    "Inference timed out after " + timeoutSeconds + "s. "
                    + "The model may be overloaded — please try again later.");
        }

        int exitCode = process.exitValue();
        if (exitCode != 0) {
            String detail = stderr.isBlank() ? "(no stderr output)" : stderr;
            log.error("Python exited with code {}: {}", exitCode, detail);
            throw new PredictException(502,
                    "Python inference failed (exit code " + exitCode + "). "
                    + "stderr: " + detail);
        }

        // Extract the last JSON line from stdout (skip log lines)
        String[] lines = stdout.split("\n");
        String jsonLine = null;
        for (int i = lines.length - 1; i >= 0; i--) {
            String line = lines[i].trim();
            if (line.startsWith("{")) {
                jsonLine = line;
                break;
            }
        }
        if (jsonLine == null) {
            throw new PredictException(502,
                    "Python produced no valid JSON output. stdout:\n" + stdout);
        }

        return jsonLine;
    }
}
