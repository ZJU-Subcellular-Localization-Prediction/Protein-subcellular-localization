package com.proteinlocal.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;

public class PredictRequest {

    @NotBlank(message = "Protein sequence must not be empty")
    @Pattern(regexp = "^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]*$",
             message = "Sequence contains invalid amino acid characters")
    private String sequence;

    public String getSequence() { return sequence; }
    public void setSequence(String sequence) { this.sequence = sequence; }
}
