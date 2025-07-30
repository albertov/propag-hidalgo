# CLAUDE.md - firelib crate

This file provides critical guidance for working with the firelib crate.

## CRITICAL CONSTRAINT: const functions and STANDARD_CATALOG

The firelib crate makes extensive use of `const fn` functions throughout its implementation. This is **absolutely critical** to maintain because it enables the STANDARD_CATALOG to be computed at compile-time as a static constant.

### Why this matters

The `Catalog::STANDARD` constant (defined in `src/lib.rs:126`) contains the standard 13 fuel models used in wildfire simulation. By using const functions throughout the codebase, Rust can evaluate all the fuel model calculations at compile time, resulting in:

1. **Zero runtime initialization cost** - The entire fuel catalog is embedded in the binary as pre-computed data
2. **No heap allocations** - Everything is stored in static memory
3. **CUDA compatibility** - Static data can be directly accessed from GPU kernels without copying

### Key const function chains

The const evaluation flows through these critical paths:

1. **Unit conversions** (`src/units.rs`) - All imperial/SI conversions are const
2. **Float operations** (`src/f32.rs`, `src/f64.rs`) - SoftFloat wrapper provides const arithmetic
3. **Particle definitions** (`src/firelib.rs:557`) - `ParticleDef::standard` is const
4. **Fuel definitions** (`src/firelib.rs:873`) - `Fuel::standard` is const
5. **Catalog creation** (`src/firelib.rs:379`) - `Catalog::make` is const

### Rules when modifying this crate

**ABSOLUTE REQUIREMENTS:**

1. **NEVER remove `const` from any function** that is part of the initialization chain for STANDARD_CATALOG
2. **NEVER introduce non-const operations** in functions called during catalog initialization
3. **ALWAYS verify that STANDARD_CATALOG remains a static constant** after any changes
4. **Test const-ness** by ensuring the crate compiles with `Catalog::STANDARD` defined as shown

### Common pitfalls to avoid

- Using `f32::sqrt()`, `f32::exp()`, etc. directly (use SoftFloat wrappers instead)
- Calling non-const trait methods in const contexts
- Using heap allocations or dynamic dispatch
- Introducing runtime conditionals in initialization paths

### Verifying const correctness

After any changes, verify that this still compiles:

```rust
impl Catalog {
    pub const STANDARD: Catalog = Catalog::make([
        // ... fuel definitions
    ]);
}
```

If you get errors about "cannot call non-const fn in const fn", you've broken the const chain.

### Float operations in const context

The crate uses a `SoftFloat` wrapper to enable floating-point operations in const contexts:

```rust
// CORRECT - const-compatible
let x = SoftFloat(1.0).mul(SoftFloat(2.0)).to_float();

// INCORRECT - not const-compatible
let x = 1.0 * 2.0;  // This won't work in const fn!
```

Always use the SoftFloat arithmetic methods (add, sub, mul, div, sqrt, exp, powf, powi) when working in const functions.