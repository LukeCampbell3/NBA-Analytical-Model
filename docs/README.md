# Documentation Index

Welcome to the NBA Player Performance Prediction System documentation.

## Getting Started

New to the system? Start here:

1. **[Main README](../README.md)**: Project overview, installation, and quick start
2. **[Configuration Guide](../configs/README.md)**: Configure the system for your needs
3. **[CLI Documentation](CLI.md)**: Learn command-line interface commands
4. **[API Documentation](API.md)**: REST API reference and examples

## Documentation Files

### User Guides

- **[README.md](../README.md)**: Main project documentation
  - Overview and key capabilities
  - Installation instructions
  - Quick start guide
  - Configuration overview
  - Development guidelines

- **[CLI Documentation](CLI.md)**: Command-line interface guide
  - All CLI commands with examples
  - Options and parameters
  - Workflow examples
  - Troubleshooting

- **[API Documentation](API.md)**: REST API reference
  - Endpoint descriptions
  - Request/response formats
  - Authentication (future)
  - Code examples in Python and JavaScript
  - Error handling

### Configuration

- **[Configuration Guide](../configs/README.md)**: Comprehensive configuration reference
  - All configuration options explained
  - Default values and recommendations
  - Environment variables
  - Best practices
  - Troubleshooting

### Implementation Details

- **[Error Handling & Logging](error_handling_logging.md)**: Error handling patterns
  - Custom exception classes
  - Structured logging
  - Error recovery strategies

- **[Matchup Constraints](matchup_constraints_implementation.md)**: Matchup constraint system
  - Opponent scheme constraints
  - Role-based bounds
  - Frontier constraints

- **[Parallelization](parallelization_implementation.md)**: Parallel processing
  - Multi-process simulation
  - Parallel benchmarking
  - Performance optimization

## Quick Reference

### Common Tasks

**Run a simulation**:
```bash
python -m src.cli.main simulate-global \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL \
  --output-json
```

**Start API server**:
```bash
uvicorn src.api.server:app --reload
```

**Run benchmarks**:
```bash
python -m src.cli.main benchmark \
  --window rolling_30_games \
  --models all \
  --output-pdf
```

**Train models**:
```bash
python -m src.cli.main train-local --event-type all
python -m src.cli.main baselines-train --model-type all
```

### Configuration Files

- `configs/default.yaml`: Main configuration file
- `configs/README.md`: Configuration documentation

### Example Scripts

- `examples/api_demo.py`: API usage examples
- `examples/matchup_constraints_demo.py`: Matchup constraints demo
- `examples/parallelization_demo.py`: Parallelization examples
- `examples/reporting_demo.py`: Report generation examples

### Test Fixtures

- `fixtures/toy_game_inputs.json`: Sample game data
- `fixtures/small_eval_window.parquet`: Sample evaluation data

## Architecture Overview

```
System Components:
├── Data Layer: CSV loading and validation
├── Feature Engineering: Rolling windows, posteriors
├── Frontiers: Efficiency frontier fitting
├── Regions: Capability region construction
├── Simulation: Markov-MC global simulator
├── Local Models: Event-specific predictions
├── Baselines: Traditional ML models
├── Calibration: Probability calibration
├── Benchmarking: Model comparison
├── Reporting: PDF/JSON/CSV reports
├── API: REST endpoints
└── CLI: Command-line interface
```

## Key Concepts

### Capability Regions

Geometric regions representing feasible player performance space:
- **Ellipsoid**: Credible region from posterior distribution
- **Polytope**: Halfspace constraints from frontiers and matchups
- **Intersection**: Final capability region

### Global Simulation

Markov-Monte Carlo simulation with game states:
- **States**: Normal, Hot, Cold, FoulRisk, WindDown
- **Transitions**: Markov chain between states
- **Offsets**: State-specific performance adjustments

### Local Models

Event-specific logistic regression:
- **Rebound Model**: Rebound probability
- **Assist Model**: Assist probability
- **Shot Model**: Shot success probability

### Blending

Combines global and local predictions:
- **Weighted**: Linear combination with configurable weights
- **Bootstrap**: Resampling-based blending
- **Recalibration**: Maintains proper uncertainty

## API Quick Start

### Health Check

```bash
curl http://localhost:8000/health
```

### Run Simulation

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d @request.json
```

### Interactive Docs

Visit http://localhost:8000/docs for interactive API documentation.

## CLI Quick Start

### Get Help

```bash
python -m src.cli.main --help
python -m src.cli.main simulate-global --help
```

### Full Pipeline

```bash
python -m src.cli.main full-pipeline \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL
```

## Development

### Running Tests

```bash
pytest tests/
pytest --cov=src tests/
```

### Code Style

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Adding Documentation

When adding new features:
1. Update relevant documentation files
2. Add docstrings to all public functions/classes
3. Update CLI help text if adding commands
4. Update API documentation if adding endpoints
5. Add examples to demonstrate usage

## Support

### Getting Help

1. Check relevant documentation file
2. Review configuration in `configs/default.yaml`
3. Check logs in `logs/system.log`
4. Run with `--help` flag for CLI commands
5. Visit `/docs` endpoint for API documentation
6. Contact development team

### Reporting Issues

When reporting issues, include:
- Command or API call that failed
- Error message and stack trace
- Configuration settings (if relevant)
- Log file excerpt
- System information (OS, Python version)

## Contributing

See [Contributing section in main README](../README.md#contributing) for guidelines.

## Version History

### 1.0.0 (Current)

- Initial release
- Complete documentation suite
- CLI and API interfaces
- Global and local models
- Baseline models for comparison
- Comprehensive benchmarking
- PDF/JSON/CSV reporting

## License

Proprietary - All rights reserved

---

**Last Updated**: 2024-01-15

**Documentation Version**: 1.0.0
