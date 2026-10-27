"""Preserve the earlier receipt and verify the additionally delivered OOD files."""
import verify_recovery_transport_20261027 as transport

if __name__ == '__main__':
    transport.OUTPUT = transport.ROOT/transport.PREFIX/'ood_transport_verification.json'
    transport.main()
