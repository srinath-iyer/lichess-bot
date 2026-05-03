"""Public homemade-engine entrypoints."""

from lib.engine_wrapper import MinimalEngine

from engines.search import RandomMove


class ExampleEngine(MinimalEngine):
	"""Compatibility base class for existing homemade examples and tests."""

