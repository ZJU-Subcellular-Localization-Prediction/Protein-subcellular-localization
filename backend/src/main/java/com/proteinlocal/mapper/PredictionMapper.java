package com.proteinlocal.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.proteinlocal.entity.Prediction;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface PredictionMapper extends BaseMapper<Prediction> {
}
