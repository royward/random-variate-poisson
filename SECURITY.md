# Security Policy

Please don't use this code anywhere mission critical - it has not been tested anywhere near sufficiently for that.

This code does no heap allocation, uses no pointers (except as references to integers), uses no libraries and makes no system calls. The only potential security implications are if something was caused downstream by the results not being as accurate as expected.

This code is thread safe.

## Supported Versions

As the changes are all bug fixes or other improvements, there is no known reason to use anything other than the latest version.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.0   | :white_check_mark: |
