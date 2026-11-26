from app.main import SIMULATIONS, simulation_status


def test_simulation_status_exposes_measured_elapsed_time_and_rate():
    SIMULATIONS['qa-simulation'] = {
        'id': 'qa-simulation',
        'status': 'running',
        'processed': 20,
        'total': 100,
        'events': [],
        'started_at': 100.0,
        'finished_at': 105.0,
    }
    try:
        result = simulation_status('qa-simulation')
        assert result['elapsed_seconds'] == 5.0
        assert result['records_per_second'] == 4.0
        assert 'started_at' not in result
        assert 'finished_at' not in result
    finally:
        SIMULATIONS.pop('qa-simulation', None)
