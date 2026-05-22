package com.proteinlocal.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.LocalDateTime;

@TableName("sequences")
public class Sequence {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String sequenceId;
    private String rawSequence;
    private Integer sequenceLength;
    private LocalDateTime createdAt;

    public Sequence() {}

    public Sequence(String sequenceId, String rawSequence, Integer sequenceLength) {
        this.sequenceId = sequenceId;
        this.rawSequence = rawSequence;
        this.sequenceLength = sequenceLength;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getSequenceId() { return sequenceId; }
    public void setSequenceId(String sequenceId) { this.sequenceId = sequenceId; }

    public String getRawSequence() { return rawSequence; }
    public void setRawSequence(String rawSequence) { this.rawSequence = rawSequence; }

    public Integer getSequenceLength() { return sequenceLength; }
    public void setSequenceLength(Integer sequenceLength) { this.sequenceLength = sequenceLength; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
