# Arc Failure Taxonomy Data Architecture Rules

*Engineering guidelines for implementing production-grade code*

## Repository-First Development

**MUST DO:**
- Analyze the entire codebase before implementing any new component
- Study existing patterns or verl
- Extend existing abstractions rather than creating new ones
- Follow established naming conventions and directory structure
- Reuse existing configuration management and error handling patterns

**NEVER DO:**
- Start implementation without understanding existing architecture
- Create duplicate functionality when reusable components exist
- Introduce inconsistent patterns or naming conventions
- Bypass established logging, metrics, or configuration systems
- Ignore existing database connection or transaction patterns

## Reusability and Modularity

**MUST DO:**
- Design every component as a reusable building block
- Create clear interfaces with well-defined contracts
- Build components that work independently and can be composed
- Use dependency injection for testability and flexibility
- Document public APIs with clear usage contracts

**NEVER DO:**
- Write monolithic components that cannot be decomposed
- Hard-code values that should be configurable
- Create tight coupling between unrelated modules
- Mix business logic with infrastructure concerns
- Build components that assume specific deployment environments

## Code Clarity and Explicitness

**MUST DO:**
- Use descriptive, unambiguous names for variables, functions, and classes
- Write self-documenting code that explains intent clearly
- Keep functions focused on single responsibilities
- Make all configuration explicit and externally configurable
- Handle all error conditions with specific, actionable error types
- Fail fast with clear error messages when required data is missing

**NEVER DO:**
- Use abbreviations, acronyms, or unclear variable names
- Write complex logic that requires extensive comments to understand
- Create deeply nested control structures or inheritance hierarchies
- Use magic numbers, strings, or unexplained constants
- Implement silent failures or generic exception handling
- Use silent fallbacks (e.g., `dict.get(key, default)`) for critical data - if data is required, the code must raise an error

## Production-Grade Implementation

**MUST DO:**
- Implement comprehensive error handling with specific exception types
- Add structured logging with appropriate context and correlation IDs
- Include retry logic with exponential backoff for transient failures
- Design for graceful degradation and circuit breaker patterns
- Plan for horizontal scaling and load distribution from day one
- Implement proper resource cleanup and connection lifecycle management

**NEVER DO:**
- Use bare try/catch blocks or swallow exceptions silently
- Rely on print statements or ad-hoc logging approaches
- Skip input validation or parameter sanitization
- Create memory leaks or unbounded resource consumption
- Build systems that cannot be monitored or debugged in production
- Implement prototype or temporary solutions in production code

## Zero Technical Debt Policy

**MUST DO:**
- Write complete, production-ready code from the first implementation
- Include comprehensive tests covering happy path and failure scenarios
- Design APIs with backward compatibility from the beginning
- Implement proper schema migration and versioning strategies
- Document architectural decisions and operational procedures

**NEVER DO:**
- Add temporary workarounds or shortcuts with promises to fix later
- Skip testing or documentation with plans to add them eventually
- Create quick fixes that require future refactoring
- Leave unfinished features or commented-out code in the repository
- Defer performance, security, or scalability considerations

## Data Architecture Specifics

**MUST DO:**
- Design schemas with proper constraints, indexes, and validation rules
- Implement data lifecycle policies and retention strategies upfront
- Plan partitioning and sharding strategies for scale
- Use appropriate data types and enforce referential integrity
- Design for cross-region replication and disaster recovery scenarios

**NEVER DO:**
- Use generic JSONB columns for data that should be properly structured
- Create tables without primary keys, foreign keys, or proper indexes
- Mix hot and cold data access patterns in the same storage layer
- Skip database constraint enforcement or validation
- Create circular dependencies between data models or services

## API and Interface Design

**MUST DO:**
- Design APIs that maintain backward compatibility by default
- Use explicit versioning for all public interfaces
- Implement proper pagination, filtering, and sorting for list operations
- Return consistent error response formats with actionable messages
- Add rate limiting and comprehensive request validation

**NEVER DO:**
- Change existing API signatures without proper deprecation cycles
- Return different data structures for the same endpoint across versions
- Expose internal implementation details through public interfaces
- Skip authentication, authorization, or input validation on public endpoints
- Create APIs without proper documentation and usage contracts

## Observability and Operations

**MUST DO:**
- Implement distributed tracing for all cross-service operations
- Add custom metrics for business-critical flows and SLA monitoring
- Log structured data with consistent schemas and correlation identifiers
- Create comprehensive health checks and readiness probes
- Design systems for zero-downtime deployments and rollbacks

**NEVER DO:**
- Log sensitive data, credentials, or personally identifiable information
- Create metrics or alerts without clear ownership and escalation procedures
- Skip performance benchmarking for critical path operations
- Implement logging that cannot be aggregated, searched, or correlated
- Deploy systems without proper monitoring, alerting, and operational runbooks

## Security and Compliance

**MUST DO:**
- Implement defense-in-depth security strategies with multiple layers
- Use principle of least privilege for all access control decisions
- Encrypt all data at rest and in transit using industry standards
- Implement comprehensive audit logging for all sensitive operations
- Plan for compliance requirements including data retention and deletion

**NEVER DO:**
- Store secrets, credentials, or sensitive configuration in code repositories
- Skip authentication or authorization for internal service communications
- Implement custom cryptography or security protocols
- Log credentials, tokens, API keys, or other sensitive authentication data
- Create overly permissive access controls or bypass security mechanisms

## Performance and Scalability

**MUST DO:**
- Profile and measure performance before implementing optimizations
- Design systems for horizontal scaling with stateless components
- Implement appropriate caching strategies at multiple layers
- Use connection pooling and efficient resource management patterns
- Plan capacity requirements and load testing strategies from the beginning

**NEVER DO:**
- Optimize prematurely without measurement and clear performance targets
- Create single points of failure or bottlenecks in system architecture
- Ignore memory usage, garbage collection, or resource utilization patterns
- Skip load testing and capacity planning for production deployments
- Create unbounded operations or resource consumption scenarios

## Implementation Standards

**Quality Gates:**
- All components must be designed for reusability across the system
- Error handling must be comprehensive with specific exception types
- Resource management must guarantee cleanup under all conditions
- Performance characteristics must be measured and documented
- Security review must be completed for all data handling operations

**Review Criteria:**
- Functional correctness: Solves the stated problem completely
- Production readiness: Handles production traffic and failure scenarios
- Maintainability: Can be understood and modified by other engineers
- Performance: Meets documented latency and throughput requirements
- Security: Follows established security best practices and compliance requirements
- Observability: Provides comprehensive monitoring and debugging capabilities

**Deployment Requirements:**
- Comprehensive test coverage including failure and edge case scenarios
- Performance benchmarks meet or exceed production requirements
- Monitoring, alerting, and operational runbooks are complete
- Rollback and disaster recovery procedures are tested and documented
- Security scanning and vulnerability assessment are completed