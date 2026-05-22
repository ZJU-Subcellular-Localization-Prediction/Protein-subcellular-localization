package com.proteinlocal.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.proteinlocal.dto.ApiResponse;
import com.proteinlocal.dto.HistoryPage;
import com.proteinlocal.dto.PredictRequest;
import com.proteinlocal.dto.PredictResponse;
import com.proteinlocal.entity.Prediction;
import com.proteinlocal.service.PredictService;
import com.proteinlocal.service.PredictionService;
import com.proteinlocal.service.SequenceService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class PredictController {

    private final PredictService predictService;
    private final SequenceService sequenceService;
    private final PredictionService predictionService;

    public PredictController(PredictService predictService,
                             SequenceService sequenceService,
                             PredictionService predictionService) {
        this.predictService = predictService;
        this.sequenceService = sequenceService;
        this.predictionService = predictionService;
    }

    @PostMapping("/predict")
    public ApiResponse<PredictResponse> predict(@Valid @RequestBody PredictRequest request) {
        String sequence = PredictService.cleanSequence(request.getSequence());
        if (sequence.isEmpty()) {
            return ApiResponse.error(400, "Sequence is empty or contains no valid amino acids");
        }

        // Save sequence record
        String seqId = PredictService.generateSequenceId(sequence);
        sequenceService.saveIfNotExists(seqId, sequence);

        // Run inference
        PredictResponse result = predictService.predict(sequence);

        // Save prediction record
        predictionService.savePrediction(result);

        return ApiResponse.ok(result);
    }

    @GetMapping("/history")
    public ApiResponse<HistoryPage> getHistory(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {

        IPage<Prediction> result = predictionService.lambdaQuery()
                .orderByDesc(Prediction::getCreatedAt)
                .page(new Page<>(page, size));

        List<Map<String, Object>> records = result.getRecords().stream().map(p -> {
            Map<String, Object> m = new java.util.LinkedHashMap<>();
            m.put("id", p.getId());
            m.put("sequenceId", p.getSequenceId());
            m.put("predictedLocation", p.getPredictedLocation());
            m.put("locationConfidence", p.getLocationConfidence());
            m.put("predictedMembrane", p.getPredictedMembrane());
            m.put("createdAt", p.getCreatedAt());
            return m;
        }).toList();

        return ApiResponse.ok(new HistoryPage(result.getTotal(), page, size, records));
    }

    @GetMapping("/history/{id}")
    public ApiResponse<Map<String, Object>> getHistoryById(@PathVariable Long id) {
        Map<String, Object> record = predictionService.findByIdWithParsed(id);
        if (record == null) {
            return ApiResponse.error(404, "Prediction record not found");
        }
        return ApiResponse.ok(record);
    }
}
