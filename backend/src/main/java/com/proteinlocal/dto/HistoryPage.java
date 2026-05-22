package com.proteinlocal.dto;

import java.util.Map;

public class HistoryPage {

    private long total;
    private int page;
    private int size;
    private java.util.List<Map<String, Object>> records;

    public HistoryPage(long total, int page, int size, java.util.List<Map<String, Object>> records) {
        this.total = total;
        this.page = page;
        this.size = size;
        this.records = records;
    }

    public long getTotal() { return total; }
    public void setTotal(long total) { this.total = total; }

    public int getPage() { return page; }
    public void setPage(int page) { this.page = page; }

    public int getSize() { return size; }
    public void setSize(int size) { this.size = size; }

    public java.util.List<Map<String, Object>> getRecords() { return records; }
    public void setRecords(java.util.List<Map<String, Object>> records) { this.records = records; }
}
