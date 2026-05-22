package com.proteinlocal.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.proteinlocal.dto.PredictResponse;
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

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${predict.python.path:python}")
    private String pythonPath;

    @Value("${predict.python.script:python/predict.py}")
    private String scriptPath;

    @Value("${predict.python.timeout-seconds:60}")
    private int timeoutSeconds;

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
        } catch (Exception e) {
            log.error("Python inference failed for sequence {}", sequenceId, e);
            throw new RuntimeException("Inference failed: " + e.getMessage(), e);
        }
    }

    private String runPythonInference(String sequence) throws Exception {
        File scriptFile = new File(scriptPath);
        if (!scriptFile.exists()) {
            scriptFile = new File("../" + scriptPath);
        }
        String absScript = scriptFile.getAbsolutePath();

        ProcessBuilder pb = new ProcessBuilder(
                pythonPath, absScript, "--sequence", sequence
        );
        pb.directory(scriptFile.getParentFile());
        pb.redirectErrorStream(false);

        log.info("Running: {} {} --sequence <SEQ>", pythonPath, absScript);

        Process process = pb.start();

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
            throw new RuntimeException("Python inference timed out after " + timeoutSeconds + "s");
        }

        int exitCode = process.exitValue();
        if (exitCode != 0) {
            throw new RuntimeException("Python exited with code " + exitCode + ": " + stderr);
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
            throw new RuntimeException("No JSON found in Python stdout. Output:\n" + stdout);
        }

        return jsonLine;
    }
}
