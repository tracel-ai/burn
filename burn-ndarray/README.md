# Burn NdArray

> [Burn](https://github.com/burn-rs/burn) ndarray backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-ndarray.svg)](https://crates.io/crates/burn-ndarray)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/burn-rs/burn-ndarray/blob/master/README.md)


## Features

This crate can be used without the standard library (`#![no_std]`) by disabling
the default `std` feature.

The following flags support various BLAS options:
* `blas-netlib` 
* `blas-openblas`
* `blas-openblas-system`
