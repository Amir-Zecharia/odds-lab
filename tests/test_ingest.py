from src.ingest import normalize_season

OLD_FORMAT = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,BbAvH,BbAvD,BbAvA,BbMxH,BbMxD,BbMxA
E0,14/08/05,Arsenal,Newcastle,2,0,H,1.5,3.6,7.0,1.52,3.55,6.8,1.6,3.8,7.5
E0,14/08/05,Middlesboro,Liverpool,0,0,D,3.2,3.2,2.2,3.1,3.15,2.25,3.3,3.3,2.4
"""

NEW_FORMAT = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSH,PSD,PSA,AvgH,AvgD,AvgA,MaxH,MaxD,MaxA,PSCH,PSCD,PSCA
E0,12/08/2023,12:30,Burnley,Man City,0,3,A,7.5,5.0,1.4,7.8,5.1,1.42,7.4,4.9,1.41,8.0,5.25,1.45,7.9,5.2,1.4
"""

RAGGED = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A
E0,20/08/00,Charlton,Man City,4,0,H,2.2,3.25,3.1
E0,20/08/00,Chelsea,West Ham,4,2,H,1.6,3.6,5.5,,,,,
,,,,,,,,,
"""


def test_old_format_bb_columns_become_avg_max():
    out = normalize_season(OLD_FORMAT, "2005-06")
    assert {"AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA"} <= set(out.columns)
    assert out.loc[0, "AvgH"] == 1.52
    assert out.loc[0, "MaxA"] == 7.5


def test_team_names_canonicalized():
    out = normalize_season(OLD_FORMAT, "2005-06")
    assert "Middlesbrough" in set(out["home_team"])
    assert "Middlesboro" not in set(out["home_team"])


def test_new_format_keeps_closing_odds():
    out = normalize_season(NEW_FORMAT, "2023-24")
    assert out.loc[0, "PSCH"] == 7.9
    assert out.loc[0, "result"] == "A"
    assert out.loc[0, "date"].year == 2023


def test_aggregates_computed_when_missing():
    out = normalize_season(RAGGED, "2000-01")
    # Only one book -> Avg == Max == that book's odds.
    assert out.loc[0, "AvgH"] == out.loc[0, "MaxH"] == 2.2


def test_ragged_and_blank_rows_survive_parsing():
    out = normalize_season(RAGGED, "2000-01")
    assert len(out) == 2
    assert out["home_goals"].tolist() == [4, 4]


def test_rows_sorted_by_date():
    reversed_rows = OLD_FORMAT.replace(
        "14/08/05,Arsenal", "20/08/05,Arsenal"
    )
    out = normalize_season(reversed_rows, "2005-06")
    assert out["date"].is_monotonic_increasing
    assert out.loc[0, "home_team"] == "Middlesbrough"
