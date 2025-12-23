-- Competitor Monitor - Supabase Schema
-- Run this in Supabase SQL Editor to create all tables

-- ============================================================================
-- RUNS TABLE - Track data collection runs
-- ============================================================================
CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    finished_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'running',
    notes TEXT,
    report_path TEXT
);

-- ============================================================================
-- ARTICLES TABLE - Raw articles from RSS feeds
-- ============================================================================
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    competitor_id VARCHAR(50),
    source_label VARCHAR(100),
    title TEXT,
    url TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    raw_snippet TEXT,
    hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint on URL to prevent duplicates
    CONSTRAINT articles_url_unique UNIQUE (url)
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_articles_competitor ON articles(competitor_id);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_run ON articles(run_id);

-- ============================================================================
-- INTEL TABLE - Analyzed intelligence from articles  
-- ============================================================================
CREATE TABLE IF NOT EXISTS intel (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    summary TEXT,
    category VARCHAR(50),
    impact_score FLOAT,
    relevance_score FLOAT,
    novelty_score FLOAT DEFAULT 0.5,
    source_count INTEGER DEFAULT 1,
    related_urls_json TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- One intel per article
    CONSTRAINT intel_article_unique UNIQUE (article_id)
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_intel_impact ON intel(impact_score DESC);
CREATE INDEX IF NOT EXISTS idx_intel_category ON intel(category);

-- ============================================================================
-- ANNOTATIONS TABLE - Domain-specific analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS annotations (
    id SERIAL PRIMARY KEY,
    intel_id INTEGER REFERENCES intel(id) ON DELETE CASCADE,
    agent_role VARCHAR(50),
    so_what TEXT,
    risk_opportunity VARCHAR(20),
    priority VARCHAR(20),
    suggested_action TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- REPORTS TABLE - Generated reports
-- ============================================================================
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- HELPER VIEW - Intel with article info (for easy querying)
-- ============================================================================
CREATE OR REPLACE VIEW intel_with_articles AS
SELECT 
    i.id,
    i.article_id,
    i.summary,
    i.category,
    i.impact_score,
    i.relevance_score,
    i.novelty_score,
    i.source_count,
    i.related_urls_json,
    i.created_at as intel_created_at,
    a.competitor_id,
    a.title,
    a.url,
    a.published_at,
    a.source_label
FROM intel i
JOIN articles a ON i.article_id = a.id
WHERE a.published_at >= '2025-01-01'
ORDER BY i.impact_score DESC;

-- ============================================================================
-- ROW LEVEL SECURITY (Optional - enable if needed)
-- ============================================================================
-- ALTER TABLE runs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE articles ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE intel ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================
-- Get top intel:
-- SELECT * FROM intel_with_articles LIMIT 50;

-- Get Tubi-specific intel:
-- SELECT * FROM intel_with_articles 
-- WHERE title ILIKE '%tubi%' OR summary ILIKE '%tubi%'
-- ORDER BY impact_score DESC;

-- Get intel by competitor:
-- SELECT * FROM intel_with_articles WHERE competitor_id = 'netflix';

