Contributing
============

We welcome contributions to CausalBoundingEngine! 

Quick Start
-----------

1. **Fork & Clone**
   
   Fork the repository on GitHub and clone your fork:

   .. code-block:: bash

      git clone https://github.com/yourusername/CausalBoundingEngine.git
      cd CausalBoundingEngine

2. **Install for Development**

   .. code-block:: bash

      pip install -e .
      pip install pytest  # For running tests

3. **Make Your Changes**

   - Create a new branch: ``git checkout -b feature-name``
   - Make your changes
   - Add tests if needed
   - Run tests: ``pytest``

4. **Submit Pull Request**

   - Push your branch: ``git push origin feature-name``
   - Open a Pull Request on GitHub
   - Describe your changes clearly

That's it! We'll review your PR and provide feedback.

Development Guidelines
----------------------

**Code Style**
   - Follow existing code patterns
   - Add docstrings to new functions/classes
   - Keep changes focused and minimal

**Testing**
   - Add tests for new functionality
   - Ensure existing tests pass: ``pytest``
   - Test with different scenarios when possible

**Documentation**
   - Update relevant documentation
   - Add examples for new features
   - Update algorithm references if adding new methods

Types of Contributions
----------------------

**Bug Fixes**
   - Report bugs via GitHub issues
   - Include minimal reproduction example
   - Fix and submit PR with test case

**New Algorithms**
   - Inherit from ``Algorithm`` base class
   - Implement ``_compute_ATE`` and/or ``_compute_PNS``
   - Add to appropriate scenario's ``AVAILABLE_ALGORITHMS``
   - Include proper citations and references

**New Scenarios**
   - Inherit from ``Scenario`` base class
   - Define ``AVAILABLE_ALGORITHMS`` mapping
   - Add to scenarios module
   - Update documentation

**Documentation**
   - Fix typos, improve clarity
   - Add examples and use cases
   - Update references and citations

Questions?
----------

- Open a GitHub issue for bugs or feature requests
- Check existing issues before creating new ones
- Be clear and provide examples when possible

Thanks for contributing! 🎉

Areas needing optimization:

1. **Large dataset handling**: Memory-efficient algorithms
2. **Parallel computation**: Multi-core algorithm implementations
3. **Approximate algorithms**: Fast approximations for exploration
4. **Caching**: Intelligent result caching

Getting Help
------------

Where to Get Help
~~~~~~~~~~~~~~~~~

1. **GitHub Discussions**: General questions and ideas
2. **GitHub Issues**: Bug reports and feature requests
3. **Documentation**: Check existing docs first
4. **Code Review**: Ask for feedback during PR process

Communication Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

- Be respectful and constructive
- Provide minimal reproducible examples for bugs
- Search existing issues before creating new ones
- Use descriptive titles and clear descriptions
- Follow up on your contributions

Recognition
-----------

Contributors are recognized in:

- **AUTHORS.md**: All contributors listed
- **CHANGELOG**: Major contributions noted in releases
- **Documentation**: Algorithm implementers credited in docs
- **Citations**: Academic contributors cited in papers

Thank you for contributing to CausalBoundingEngine! Your efforts help make causal inference more accessible to researchers and practitioners worldwide.
