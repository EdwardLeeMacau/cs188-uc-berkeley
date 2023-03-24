################################################################################
# Specify makeflags
################################################################################

.DEFAULT_GOAL = primary

MAKEFLAGS += -j

################################################################################
# Specify make tools and makeflags
################################################################################

CP      = cp
ZIP     = zip
ZIPINFO = zipinfo
MKDIR_P = mkdir -p $@

################################################################################
# Specify make targets
################################################################################

root = .
dir  := $(root)

TARGET_BINS :=
CLEAN_FILES :=
CLEAN_DIRS  :=

# ---------------------------------------------------------------------------- #

TARGET_BIN_$(dir) := $(dir)/r11922001_hw1.zip

$(TARGET_BIN_$(dir)): $(dir)/r11922001_hw1.py
	@printf "  %-8s$@\n" "ZIP"
	$(Q)$(ZIP) $@ $^

TARGET_BINS += $(TARGET_BIN_$(dir))
CLEAN_FILES += $(TARGET_BIN_$(dir))

# ---------------------------------------------------------------------------- #

TARGET_BIN_$(dir) := $(dir)/r11922001_hw1.py

$(TARGET_BIN_$(dir)): $(dir)/search.py
	@printf "  %-8s$@\n" "CP"
	$(Q)$(CP) $< $@

TARGET_BINS += $(TARGET_BIN_$(dir))
CLEAN_FILES += $(TARGET_BIN_$(dir))

################################################################################
# Declare make targets
################################################################################

.PHONY: primary
primary: $(TARGET_BINS)
	$(Q)$(ZIPINFO) $(dir)/r11922001_hw1.zip

.PHONY: clean
clean:
	$(Q)rm -f $(CLEAN_FILES)
	$(Q)rm -rf $(CLEAN_DIRS)

.PHONY: distclean
distclean: clean
	$(Q)find . -name "*.[ado]" -exec rm -rf {} \;

.PHONY: help
help: ;

################################################################################
# Directory generic make targets
################################################################################

%.o: %.c
	@printf "  %-8s$@\n" "CC"
	$(Q)$(CC) $(COMPILE) $(CCFLAGS)

$(bindir):
	@printf "  %-8s$@\n" "MKDIR"
	$(Q)$(MKDIR_P)

.SECONDARY: $(CLEAN_FILES)

################################################################################
# Make options
################################################################################

V ?= 0
ifeq ($(V),0)
    Q = @
else
    Q =
endif
