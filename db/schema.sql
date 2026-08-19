-- Player and team weekly performance stats
CREATE TABLE IF NOT EXISTS player_stats (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT,
    team TEXT,
    opponent TEXT,
    snap_pct REAL,
    targets INTEGER,
    receptions INTEGER,
    rec_yards INTEGER,
    rush_attempts INTEGER,
    rush_yards INTEGER,
    pass_attempts INTEGER,
    pass_yards INTEGER,
    pass_tds INTEGER,
    rush_tds INTEGER,
    rec_tds INTEGER,
    red_zone_touches INTEGER,
    air_yards INTEGER,
    fantasy_points REAL,
    PRIMARY KEY (season, week, player_id)
);

-- Injury / practice report status
CREATE TABLE IF NOT EXISTS injury_reports (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    team TEXT,
    report_status TEXT,       -- e.g. Questionable, Doubtful, Out
    practice_status TEXT,     -- e.g. DNP, Limited, Full
    injury_description TEXT,
    PRIMARY KEY (season, week, player_id)
);

-- Vegas lines and implied team totals
CREATE TABLE IF NOT EXISTS game_odds (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    spread REAL,               -- negative = home favored
    total REAL,                -- game over/under
    home_implied_total REAL,   -- derived: (total / 2) - (spread / 2)
    away_implied_total REAL,   -- derived: (total / 2) + (spread / 2)
    PRIMARY KEY (season, week, home_team, away_team)
);

-- Next week's matchup schedule
CREATE TABLE IF NOT EXISTS schedule (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    game_date TEXT,
    home_score INTEGER,   -- NULL until the game is played
    away_score INTEGER,   -- NULL until the game is played
    PRIMARY KEY (season, week, home_team, away_team)
);

-- Your league's actual scoring settings, one row per stat category.
-- Replaces the hardcoded SCORING dict in ingest/stats.py once populated.
CREATE TABLE IF NOT EXISTS league_settings (
    season INTEGER NOT NULL,
    stat_category TEXT NOT NULL,   -- e.g. 'passing_yards', 'receptions'
    points_per_unit REAL,           -- e.g. 0.04 for pass yards, 1.0 for PPR receptions
    PRIMARY KEY (season, stat_category)
);

-- Fantasy competitor rosters: which real players are on which fantasy team
CREATE TABLE IF NOT EXISTS rosters (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    fantasy_team_id TEXT NOT NULL,
    fantasy_team_name TEXT,
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    lineup_slot TEXT,   -- e.g. 'QB', 'BENCH', 'FLEX'
    PRIMARY KEY (season, week, fantasy_team_id, player_id)
);

-- Bridges ESPN's player IDs (used in rosters) to nflverse player IDs
-- (used in player_stats), matched by name + team since the two systems
-- don't share a common ID.
CREATE TABLE IF NOT EXISTS player_id_map (
    espn_player_id TEXT PRIMARY KEY,
    nflverse_player_id TEXT,
    player_name TEXT,
    team TEXT,
    match_confidence TEXT   -- 'exact' or 'fuzzy', flags rows worth double-checking
);

-- The actual output of feature engineering: one row per player per week,
-- ready to feed into the projection, xFP, and injury risk models.
CREATE TABLE IF NOT EXISTS features (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    opponent TEXT,
    avg_pts_l3 REAL,              -- mean fantasy points, trailing 3 weeks
    avg_pts_season REAL,          -- mean fantasy points, season-to-date
    target_share_trend REAL,      -- slope of targets over trailing 3 weeks
    snap_pct_l3 REAL,             -- mean snap pct, trailing 3 weeks
    opp_def_rank_allowed INTEGER, -- 1 = toughest matchup, 32 = easiest, by position
    implied_team_total REAL,      -- from game_odds for this week's matchup
    injury_status TEXT,           -- most recent report_status, if any
    fantasy_points REAL,          -- actual result, filled in after the week (the label)
    PRIMARY KEY (season, week, player_id)
);

-- Model output: predicted fantasy points per player for a given week.
CREATE TABLE IF NOT EXISTS projections (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    predicted_points REAL,
    PRIMARY KEY (season, week, player_id)
);

-- Team-level offensive environment for a completed season, used as a
-- proxy for "how good is this offense" going into the NEXT season
-- (since the upcoming season's own stats obviously don't exist yet).
CREATE TABLE IF NOT EXISTS team_season_stats (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    points_per_game REAL,
    points_allowed_per_game REAL,
    plays_per_game REAL,
    pass_rate REAL,          -- pass attempts / total plays
    PRIMARY KEY (season, team)
);

-- Full-season schedule difficulty per team per position, computed by
-- averaging each opponent's PRIOR season defensive rank against that
-- position across all of a team's games in the target season.
CREATE TABLE IF NOT EXISTS schedule_strength (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    position TEXT NOT NULL,
    avg_opp_def_rank REAL,   -- higher = easier full-season schedule
    PRIMARY KEY (season, team, position)
);

-- Draft-time features: one row per player per season, built entirely
-- from PRIOR data (past performance, prior team environment, upcoming
-- schedule) so it's usable before a single game of the target season
-- has been played.
CREATE TABLE IF NOT EXISTS preseason_features (
    season INTEGER NOT NULL,        -- the season being projected
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    last_season_ppg REAL,           -- avg fantasy points/game, most recent season
    two_season_avg_ppg REAL,        -- weighted avg across up to 2 prior seasons
    games_played_last_season INTEGER,
    team_proj_ppg REAL,             -- team's points_per_game, from team_season_stats
    schedule_strength_rank REAL,    -- from schedule_strength
    fantasy_points_per_game REAL,   -- LABEL: actual result, filled in after the season
    PRIMARY KEY (season, player_id)
);

-- Draft-time model output.
CREATE TABLE IF NOT EXISTS preseason_projections (
    season INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    predicted_ppg REAL,
    PRIMARY KEY (season, player_id)
);

-- How many starting slots your league requires per position - needed
-- to know how many players "count" as your starters at each position
-- when computing team needs.
CREATE TABLE IF NOT EXISTS league_roster_settings (
    season INTEGER NOT NULL,
    lineup_slot TEXT NOT NULL,   -- e.g. 'QB', 'RB', 'FLEX', 'BENCH'
    slot_count INTEGER,
    PRIMARY KEY (season, lineup_slot)
);

-- Computed team need per position: positive = weakness (recommend
-- filling it), negative = strength (well covered, lower priority).
CREATE TABLE IF NOT EXISTS team_needs (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,        -- 0 = preseason/draft-time needs
    fantasy_team_id TEXT NOT NULL,
    position TEXT NOT NULL,
    your_starter_avg REAL,        -- avg projected value of your current starters here
    replacement_baseline REAL,    -- typical rostered-player value league-wide here
    need_score REAL,              -- replacement_baseline - your_starter_avg
    PRIMARY KEY (season, week, fantasy_team_id, position)
);

-- Historical calibration: of players who carried a given injury report
-- status, what fraction actually played that week? Computed once from
-- real outcomes, reused every week rather than re-derived each time.
CREATE TABLE IF NOT EXISTS injury_risk_rates (
    report_status TEXT PRIMARY KEY,   -- e.g. 'Questionable', 'Doubtful', 'Out'
    play_probability REAL,
    sample_size INTEGER               -- how many historical cases this is based on
);

-- Weekly application: each player's play probability for the upcoming
-- week, based on their current report status and injury_risk_rates.
CREATE TABLE IF NOT EXISTS injury_risk (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    report_status TEXT,
    play_probability REAL,
    PRIMARY KEY (season, week, player_id)
);

-- Draft capital: which round/pick each player was drafted at. The
-- single strongest predictor of rookie fantasy production available
-- before they've played a snap.
CREATE TABLE IF NOT EXISTS draft_capital (
    draft_year INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    round INTEGER,
    overall_pick INTEGER,
    PRIMARY KEY (draft_year, player_id)
);

-- Fitted curve mapping draft capital to expected rookie-season points
-- per game, by position. Refit periodically as more rookie classes
-- of data become available.
CREATE TABLE IF NOT EXISTS rookie_baseline_curve (
    position TEXT PRIMARY KEY,
    intercept REAL,
    slope REAL,       -- fit on log(overall_pick): ppg = intercept + slope * ln(pick)
    sample_size INTEGER
);

-- Projected points-per-game for each team for a season, before that
-- season has been played (used for draft-time features).
CREATE TABLE IF NOT EXISTS team_projections (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    proj_ppg REAL,
    source TEXT,   -- e.g. 'shrinkage_prior_season'
    PRIMARY KEY (season, team)
);

-- How tough each team's full-season schedule is per position, based on
-- the defenses they'll face - computed before the season is played.
CREATE TABLE IF NOT EXISTS schedule_strength (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    position TEXT NOT NULL,
    avg_opp_def_rank REAL,   -- avg opponent defensive rank across the season; 1=hardest, 32=easiest
    PRIMARY KEY (season, team, position)
);

-- Season-level features for draft/preseason projections: one row per
-- player per season, combining career history, team outlook, and
-- schedule strength. actual_season_pts is NULL until that season is
-- complete (same pattern as features.fantasy_points).
CREATE TABLE IF NOT EXISTS draft_features (
    season INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    career_avg_ppg REAL,        -- recency-weighted avg points/game, up to 3 prior seasons
    last_season_total_pts REAL,
    last_season_games INTEGER,
    team_proj_ppg REAL,
    schedule_strength REAL,
    actual_season_pts REAL,     -- filled in once the season is complete (the label)
    PRIMARY KEY (season, player_id)
);

-- Draft model output: predicted full-season fantasy points per player.
CREATE TABLE IF NOT EXISTS draft_projections (
    season INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    predicted_season_pts REAL,
    PRIMARY KEY (season, player_id)
);

-- Bridges ESPN's player IDs (used in rosters) to nflverse's player IDs
-- (used in player_stats), since the two systems don't share an ID scheme.
CREATE TABLE IF NOT EXISTS player_id_map (
    espn_id TEXT PRIMARY KEY,
    nflverse_id TEXT,
    player_name TEXT,
    team TEXT,
    position TEXT,
    match_confidence TEXT   -- 'exact', 'fuzzy', or 'manual' - see build_id_map.py
);

-- Computed features per player per week, ready to feed into models.
-- This is the table step 3 (the projection model) reads from.
CREATE TABLE IF NOT EXISTS features (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id TEXT NOT NULL,        -- nflverse id (canonical)
    fantasy_points REAL,             -- this week's actual points (the label)
    rolling_avg_3wk REAL,            -- avg fantasy points, last 3 games
    season_avg REAL,                 -- avg fantasy points, season to date
    target_share REAL,               -- this week's share of team targets
    target_share_trend REAL,         -- 3-wk avg target share minus season avg
    snap_pct_trend REAL,             -- 3-wk avg snap % minus season avg
    opp_def_rank REAL,               -- opponent's rank vs this position (1=toughest)
    implied_team_total REAL,         -- from game_odds
    is_home INTEGER,                 -- 1 if home game, 0 if away
    PRIMARY KEY (season, week, player_id)
);
