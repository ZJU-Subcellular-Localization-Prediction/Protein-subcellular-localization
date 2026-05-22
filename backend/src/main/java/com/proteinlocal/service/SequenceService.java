package com.proteinlocal.service;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.proteinlocal.entity.Sequence;
import com.proteinlocal.mapper.SequenceMapper;
import org.springframework.stereotype.Service;

@Service
public class SequenceService extends ServiceImpl<SequenceMapper, Sequence> {

    public Sequence findBySequenceId(String sequenceId) {
        return lambdaQuery().eq(Sequence::getSequenceId, sequenceId).one();
    }

    public Sequence saveIfNotExists(String sequenceId, String rawSequence) {
        Sequence existing = findBySequenceId(sequenceId);
        if (existing != null) {
            return existing;
        }
        Sequence seq = new Sequence(sequenceId, rawSequence, rawSequence.length());
        save(seq);
        return seq;
    }
}
