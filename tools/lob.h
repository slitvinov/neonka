enum { nl = 8 };
struct Row {
  int16_t askRate[nl];
  int16_t bidRate[nl];
  int16_t askSize[nl];
  int16_t bidSize[nl];
  int16_t askNC[nl];
  int16_t bidNC[nl];
  int16_t y;
};

static int read_raw(const char *path, struct Row **rows, int64_t *n) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "sessions.c: error: failed to open '%s'\n", path);
    return -1;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    fprintf(stderr, "sessions.c: error: fstat failed for '%s'\n", path);
    close(fd);
    return -1;
  }
  int64_t bytes = (int64_t)st.st_size;
  if (bytes % (int64_t)sizeof(struct Row) != 0) {
    fprintf(stderr, "sessions.c: error: bad file size '%s'\n", path);
    close(fd);
    return -1;
  }
  void *map = mmap(NULL, (size_t)bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    fprintf(stderr, "sessions.c: error: mmap failed for '%s'\n", path);
    return -1;
  }
  *rows = map;
  *n = bytes / (int64_t)sizeof(struct Row);
  return 0;
}

static void free_raw(struct Row *rows, int64_t n) {
  munmap(rows, (size_t)n * sizeof(struct Row));
}
