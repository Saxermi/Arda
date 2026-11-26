CC ?= cc
PYTHON ?= python3
BLAS ?= 0
LAPACK ?= 0
SANITIZE ?= 0
LTO ?= 0
override CPPFLAGS += -D_XOPEN_SOURCE=700
CFLAGS ?= -std=c11 -O2 -g -Wall -Wextra -Wpedantic
override LDLIBS += -lm
ifeq ($(BLAS),1)
override CPPFLAGS += -DARDA_BLAS
override LDLIBS += -lblas
endif
ifeq ($(LAPACK),1)
override CPPFLAGS += -DARDA_LAPACK
override LDLIBS += -llapack -lblas
endif
ifeq ($(LTO),1)
override CFLAGS += -flto
override LDFLAGS += -flto
endif
ifeq ($(SANITIZE),1)
override CFLAGS += -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie
override LDFLAGS += -fsanitize=address,undefined -no-pie
endif

BUILD := build/blas$(BLAS)-lapack$(LAPACK)-sanitize$(SANITIZE)$(if $(filter 1,$(LTO)),-lto)
SRC := $(wildcard src/*.c)
OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))
BIN := $(BUILD)/arda
STAMP := $(BUILD)/flags.txt
GENERATED := src/ode_source.h src/opt_source.h src/numerics_source.h src/uncertainty_source.h
.PHONY: all arda test test-reference sanitize clean FORCE check-generated format format-check
all: arda
arda: $(BIN)
	ln -sfn $(BIN) $@

# Track compiler/flag changes even when the source files did not change.
$(STAMP): FORCE
	@mkdir -p $(@D)
	@printf '%s\n' '$(CC)' '$(CPPFLAGS)' '$(CFLAGS)' '$(LDFLAGS)' '$(LDLIBS)' > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

$(BUILD)/%.o: src/%.c $(STAMP)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/vm.o: $(GENERATED)
$(BIN): $(OBJ) $(STAMP)
	$(CC) $(OBJ) $(LDFLAGS) $(LDLIBS) -o $@
-include $(OBJ:.o=.d)

src/ode_source.h: lib/ode.arda tools/embed_ode.py tools/embed_module.py
	$(PYTHON) tools/embed_ode.py
src/opt_source.h: lib/opt.arda tools/embed_opt.py tools/embed_module.py
	$(PYTHON) tools/embed_opt.py
src/numerics_source.h: lib/numerics.arda tools/embed_numerics.py tools/embed_module.py
	$(PYTHON) tools/embed_numerics.py
src/uncertainty_source.h: lib/uncertainty.arda tools/embed_uncertainty.py tools/embed_module.py
	$(PYTHON) tools/embed_uncertainty.py
check-generated:
	$(PYTHON) tools/embed_ode.py --check
	$(PYTHON) tools/embed_opt.py --check
	$(PYTHON) tools/embed_numerics.py --check
	$(PYTHON) tools/embed_uncertainty.py --check

test: arda
	ARDA_BIN="$(abspath $(BIN))" $(PYTHON) -m unittest discover -s tests -v
# Optional dependencies: numpy and scipy, installed in the development environment.
test-reference: arda
	ARDA_BIN="$(abspath $(BIN))" $(PYTHON) -m unittest discover -s tests -p 'reference_*.py' -v
sanitize:
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 $(MAKE) SANITIZE=1 test
format:
	clang-format -i $(filter-out $(GENERATED),$(wildcard src/*.c src/*.h))
format-check:
	clang-format --dry-run --Werror $(filter-out $(GENERATED),$(wildcard src/*.c src/*.h))
clean:
	rm -rf build arda
