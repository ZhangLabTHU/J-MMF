# ── J-MMF Makefile ──
# Usage:
#   make release          Build release archives from current HEAD
#   make clean-release    Remove generated archives

VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null || echo "v1.0.0")
NAME    := J-MMF
TAG     := $(VERSION)

.PHONY: release clean-release

release:
	@echo "Building release archives for $(TAG)..."
	git archive --format=tar.gz --prefix=$(NAME)-$(TAG)/ -o $(NAME)-$(TAG).tar.gz $(TAG)
	git archive --format=zip --prefix=$(NAME)-$(TAG)/ -o $(NAME)-$(TAG).zip $(TAG)
	@echo "Done:"
	@ls -lh $(NAME)-$(TAG).tar.gz $(NAME)-$(TAG).zip

clean-release:
	rm -f $(NAME)-*.tar.gz $(NAME)-*.zip