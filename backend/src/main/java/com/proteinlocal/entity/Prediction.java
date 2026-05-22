package com.proteinlocal.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.LocalDateTime;

@TableName("predictions")
public class Prediction {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String sequenceId;
    private String predictedLocation;
    private Double locationConfidence;
    private String predictedMembrane;
    private Double membraneConfidence;

    @TableField("all_probabilities")
    private String allProbabilities;

    @TableField("attention_data")
    private String attentionData;

    private String modelVersion;
    private Integer inferenceTimeMs;
    private LocalDateTime createdAt;

    public Prediction() {}

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

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

    public String getAllProbabilities() { return allProbabilities; }
    public void setAllProbabilities(String allProbabilities) { this.allProbabilities = allProbabilities; }

    public String getAttentionData() { return attentionData; }
    public void setAttentionData(String attentionData) { this.attentionData = attentionData; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }

    public Integer getInferenceTimeMs() { return inferenceTimeMs; }
    public void setInferenceTimeMs(Integer inferenceTimeMs) { this.inferenceTimeMs = inferenceTimeMs; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
