# Loopback TLS test material

The archived measurement used a fixed self-signed RSA-2048 certificate solely
for the reproducible `127.0.0.1` TLS framing benchmark. The public repository
does not ship its private key. `system.tls_test_material` instead generates an
equivalent disposable certificate/key pair in a temporary directory for each
reproduction run, and the directory is removed afterward.

Client certificate verification remains disabled by design because the
experiment measures persistent TLS 1.3 transport/framing cost rather than PKI
deployment or certificate validation. Generated material must never be reused
outside this loopback test.
