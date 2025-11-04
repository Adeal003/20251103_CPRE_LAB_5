.PHONY: build clean run depend check_update pull_update submit format help
.SUFFIXES: .o
.SECONDARY:

# Recursive Wildcard (https://stackoverflow.com/questions/2483182/recursive-wildcards-in-gnu-make/18258352#18258352)
rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))
GCCVERSIONLTQ9 := $(shell expr `gcc -dumpversion | cut -f1 -d.` \< 9)

DEBUG ?= true
SIMD ?= false
BDIR = build
BIN = $(BDIR)
SDIR = src
EXE = $(BIN)/ml
EXE_DEBUG = $(BIN)/ml_debug

# ifeq ($(OS), Windows_NT) # Windows
# 	CC_Linux = 
# 	CC_WIN = cl
# 	WIN_SETUP = vcvarsall.bat amd64
#	EXECUTE_EXT = .exe
# else # Linux
	CC_LINUX = g++ 
	CC_WIN = x86_64-w64-mingw32-g++ -static-libgcc -static-libstdc++ -fstack-protector
	CC_ALL = -lstdc++ -Wall -Werror -pedantic -std=c++11
	CC_DEBUG = -g -Og
	CC_OPT_FLAGS = -O3 -fno-tree-pre
	CC_SIMD_FLAGS = -march=native
	CC_FLAGS = $(CC_ALL) $(CC_OPT_FLAGS) $(if $(filter $(SIMD), true), $(CC_SIMD_FLAGS),)
	CC_DEBUG_FLAGS = $(CC_ALL) $(CC_DEBUG) $(if $(filter $(SIMD), true), $(CC_SIMD_FLAGS),)

# Libs
# CC_FLAGS_END += -pthread -lfmt
CC_FLAGS_END += -pthread

#	Check if GCC is version 9 or greater, and if not, include an additional flag for the filesystem lib
#	ifeq "$(GCCVERSIONLTQ9)" "1"
#    	CC_FLAGS_END += -lstdc++fs
#	endif
# endif

INC = -Iinc
SOURCE_FILES = $(call rwildcard, $(SDIR), *.cpp) $(call rwildcard, $(SDIR), *.c)	# Find all source files
HEADER_FILES = $(call rwildcard, $(SDIR), *.h)  $(call rwildcard, $(SDIR), *.hpp)	# Find all header files
DEPEND_FILES = $(call rwildcard, $(BDIR), *.d)

_OBJS = $(patsubst %.cpp, %.o, $(SOURCE_FILES))		# Calculate names of object files by replacing .c and .cpp with .o
OBJS = $(patsubst $(SDIR)/%, $(BDIR)/%, $(_OBJS))	# Create paths for those names by appending the build dir
_OBJS_DEBUG = $(patsubst %.cpp, %_debug.o, $(SOURCE_FILES))		# Calculate names of object files by replacing .c and .cpp with .o
OBJS_DEBUG = $(patsubst $(SDIR)/%, $(BDIR)/%, $(_OBJS_DEBUG))	# Create paths for those names by appending the build dir


# -include $(DEPEND_FILES)

# Debug Prints
# $(warning SOURCE_FILES: $(SOURCE_FILES))
# $(warning HEADER_FILES: $(HEADER_FILES))
# $(warning _OBJS: $(_OBJS))
# $(warning OBJS: $(OBJS))
# $(warning EXE: $(EXE))
# $(warning DEBUG: $(DEBUG))


# Make Rules
all: build 
update: check_update pull_update
submit: check_update format build zip

# Build compiler from source
build: dir_struct $(EXE)
rebuild: clean build

build_debug: dir_struct $(EXE_DEBUG)
redebug: clean build_debug

# Generate object files
$(BIN)/ml: $(OBJS)
	$(CC_LINUX) $(CC_FLAGS) $(OBJS) -o $@ $(CC_FLAGS_END)

$(BIN)/ml_debug: $(OBJS_DEBUG)
	$(CC_LINUX) $(CC_FLAGS_DEBUG) $(OBJS_DEBUG) -o $@ $(CC_FLAGS_END)

$(BDIR)/%.o: $(SDIR)/%.cpp
	mkdir -p $(dir $@)
	$(CC_LINUX) $(CC_FLAGS) -c $(INC) -o $@ $< $(CFLAGS)

$(BDIR)/%_debug.o: $(SDIR)/%.cpp
	mkdir -p $(dir $@)
	$(CC_LINUX) $(CC_FLAGS_DEBUG) -c $(INC) -o $@ $< $(CFLAGS)

# Run the framework
#run:
	#@shift;
	#./$(BDIR)/ml $@


#debug:
	#@shift;
	#./$(BDIR)/ml_debug $@


# Update the framework
check_update:
	@git remote add framework-upstream https://git.ece.iastate.edu/dwyer/cpre487-587-lab2.git >/dev/null 2>&1 || true
	@git fetch upstream

pull_update:
	@git pull --no-rebase framework-upstream master

submit: clean format build zip
	@echo "Submission Ziped: Submission.zip"

zip:
	zip -9 -x Makefile -x "*.o" -r Submission.zip .

help:
	@echo -e "487/587 ML Framework Help:\n" \
	      "\tUsage: make <target> [params]\n" \
	      "\n" \
	      "Targets:\n" \
	      "\tbuild: \t\tBuilds the framework (same as 'make')\n" \
	      	"\t\t\tNOTE: header files (.h/.hpp) changes are not detected, please 'clean' first\n" \
	      "\trebuild: \tPerforms a 'clean' then 'build'\n" \
	      "\tbuild_debug: \tSame as 'build', but with without optimizations and debug information\n" \
	      "\tredebug: \tPerforms a 'clean' then 'debug' build\n" \
	      "\tclean: \t\tCleans all build artifacts\n" \
	      "\tformat: \tFormats all source files" \
	      "\tupdate: \tChecks for a framework update. If one is found, it is pulled\n" \
	      "\tsubmit: \tZips directory for submission\n" \
	      "\thelp: \t\tShows this help message"
#"\trun: \t\tRuns the optimized binary with the provided params\n" \
#"\tdebug: \t\tRuns the debug binary with the provided params\n" \ 

# Format source files
format:
	@clang-format -i --verbose -Werror -style=file $(SOURCE_FILES) $(HEADER_FILES)

# Clean build files
clean:
	@echo "Deleting $(shell test -d $(BDIR) && find .//$(BDIR) ! -name . -print | grep -c // || echo "0") Files in: ./$(BDIR)/"
	@rm -rf $(BDIR)/

# Generate the build folder structure
dir_struct:
	@mkdir -p $(BDIR)/

