import syncscope.cli as cli


def test_demo_runs_and_returns_zero(capsys):
    rc = cli.main(["demo"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sync:" in out
    assert "asd:" in out


def test_no_command_prints_help_and_fails(capsys):
    rc = cli.main([])
    captured = capsys.readouterr().out
    assert rc == 1
    assert "usage" in captured.lower()


def test_sync_without_extra_reports_missing_dependency(capsys):
    # librosa/opencv are not installed in the test environment, so the command
    # should fail gracefully with a helpful message rather than a traceback.
    rc = cli.main(["sync", "missing.mp4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "extra" in out
