package com.proteinlocal.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.proteinlocal.dto.PredictResponse;
import com.proteinlocal.entity.Prediction;
import com.proteinlocal.mapper.PredictionMapper;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class PredictionService extends ServiceImpl<PredictionMapper, Prediction> {

    private final ObjectMapper objectMapper = new ObjectMapper();

    public void savePrediction(PredictResponse resp) {
        Prediction pred = new Prediction();
        pred.setSequenceId(resp.getSequenceId());
        pred.setPredictedLocation(resp.getPredictedLocation());
        pred.setLocationConfidence(resp.getLocationConfidence());
        pred.setPredictedMembrane(resp.getPredictedMembrane());
        pred.setMembraneConfidence(resp.getMembraneConfidence());
        pred.setModelVersion(resp.getModelVersion());
        pred.setInferenceTimeMs(resp.getInferenceTimeMs());

        try {
            if (resp.getAllProbabilities() != null) {
                pred.setAllProbabilities(objectMapper.writeValueAsString(resp.getAllProbabilities()));
            }
            if (resp.getAttentionWeights() != null) {
                pred.setAttentionData(objectMapper.writeValueAsString(resp.getAttentionWeights()));
            }
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize prediction data", e);
        }

        save(pred);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> findByIdWithParsed(Long id) {
        Prediction pred = getById(id);
        if (pred == null) return null;

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("id", pred.getId());
        result.put("sequenceId", pred.getSequenceId());
        result.put("predictedLocation", pred.getPredictedLocation());
        result.put("locationConfidence", pred.getLocationConfidence());
        result.put("predictedMembrane", pred.getPredictedMembrane());
        result.put("membraneConfidence", pred.getMembraneConfidence());
        result.put("modelVersion", pred.getModelVersion());
        result.put("inferenceTimeMs", pred.getInferenceTimeMs());
        result.put("createdAt", pred.getCreatedAt());

        try {
            if (pred.getAllProbabilities() != null) {
                result.put("allProbabilities", objectMapper.readValue(pred.getAllProbabilities(), Map.class));
            }
        } catch (JsonProcessingException ignored) {}

        return result;
    }
}
