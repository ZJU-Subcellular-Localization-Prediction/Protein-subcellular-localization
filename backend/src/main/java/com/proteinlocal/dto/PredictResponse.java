package com.proteinlocal.dto;

import java.util.Map;

public class PredictResponse {

    private String sequenceId;
    private String predictedLocation;
    private Double locationConfidence;
    private String predictedMembrane;
    private Double membraneConfidence;
    private Map<String, Double> allProbabilities;
    private double[][] attentionWeights;
    private String modelVersion;
    private Integer inferenceTimeMs;

    public String getSequenceId() { return sequenceId; }
    public void setSequenceId(String sequenceId) { this.sequenceId = sequenceId; }

    public String getPredictedLocation() { return predictedLocation; }
    public void setPredictedLocation(String predictedLocation) { this.predictedLocation = predictedLocation; }

    public Double getLocationConfidence() { return locationConfidence; }
    public void setLocationConfidence(Double locationConfidence) { this.locationConfidence = locationConfidence; }

    public String getPredictedMembrane() { return predictedMembrane; }
    public void setPredictedMembrane(String predictedMembrane) { this.predictedMembrane = predictedMembrane; }

    public Double getMembraneConfidence() { return membraneConfidence; }
    public void setMembraneConfidence(Double membraneConfidence) { this.membraneConfidence = membraneConfidence; }

    public Map<String, Double> getAllProbabilities() { return allProbabilities; }
    public void setAllProbabilities(Map<String, Double> allProbabilities) { this.allProbabilities = allProbabilities; }

    public double[][] getAttentionWeights() { return attentionWeights; }
    public void setAttentionWeights(double[][] attentionWeights) { this.attentionWeights = attentionWeights; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }

    public Integer getInferenceTimeMs() { return inferenceTimeMs; }
    public void setInferenceTimeMs(Integer inferenceTimeMs) { this.inferenceTimeMs = inferenceTimeMs; }
}
