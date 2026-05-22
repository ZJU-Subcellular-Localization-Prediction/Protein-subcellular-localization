package com.proteinlocal.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class PredictResponse {

    @JsonProperty("sequence_id")
    private String sequenceId;

    @JsonProperty("predicted_location")
    private String predictedLocation;

    @JsonProperty("location_confidence")
    private Double locationConfidence;

    @JsonProperty("predicted_membrane")
    private String predictedMembrane;

    @JsonProperty("membrane_confidence")
    private Double membraneConfidence;

    @JsonProperty("all_probabilities")
    private Map<String, Double> allProbabilities;

    @JsonProperty("attention_weights")
    private double[][] attentionWeights;

    @JsonProperty("model_version")
    private String modelVersion;

    @JsonProperty("inference_time_ms")
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
