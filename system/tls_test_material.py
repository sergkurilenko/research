"""Create disposable TLS material for loopback-only reproducibility tests.

The benchmark disables certificate verification because it measures local
framing and transport overhead, not authentication. Generating the key in a
temporary directory keeps private-key material out of the public repository.
"""

from __future__ import annotations

import ipaddress
from datetime import datetime, timezone
from pathlib import Path


def generate_loopback_tls_material(directory: Path) -> tuple[Path, Path]:
    """Generate a temporary RSA certificate/key pair for ``localhost``."""

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:  # pragma: no cover - minimal environments only
        raise RuntimeError(
            "cryptography is required to generate disposable loopback TLS material"
        ) from exc

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    certificate_path = directory / "localhost_cert.pem"
    private_key_path = directory / "localhost_key.pem"

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]
    )
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime(2020, 1, 1, tzinfo=timezone.utc))
        .not_valid_after(datetime(2040, 1, 1, tzinfo=timezone.utc))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return certificate_path, private_key_path
