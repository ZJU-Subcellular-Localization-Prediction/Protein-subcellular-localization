package com.proteinlocal.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.proteinlocal.entity.Sequence;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface SequenceMapper extends BaseMapper<Sequence> {
}
