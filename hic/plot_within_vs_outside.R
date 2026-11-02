#!/usr/bin/env Rscript
# Plots for hic/within_vs_outside.py: Hi-C contacts inside vs outside the bin.
#
# Rscript plot_within_vs_outside.R --dir hic_within/ [--sample ZymoFecal] [--min-contacts 10]
#   A  per binned contig: contacts within its bin vs contacts with other bins, per epirefine decision
#   B  per binned contig: within-bin share against epirefine's P(belongs to its bin)
#   C  per bin: within-bin share of each refined bin set against the original
# Writes within_vs_outside.pdf / .png into --dir.

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(patchwork); library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
opt <- list(sample = "", `min-contacts` = "10")
key <- NULL
for (a in args) {
  if (startsWith(a, "--")) { key <- sub("^--", "", a); opt[[key]] <- character(0) }
  else opt[[key]] <- c(opt[[key]], a)
}
dir <- opt$dir
minc <- as.numeric(opt$`min-contacts`)
INK <- "#0b0b0b"; INK2 <- "#52514e"; BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; GREY <- "#b8b7b0"
theme_set(theme_minimal(base_size = 10) +
          theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#e6e5e0", linewidth = 0.3),
                plot.title = element_text(face = "bold", colour = INK), plot.subtitle = element_text(colour = INK2),
                strip.text = element_text(face = "bold", colour = INK), axis.text = element_text(colour = INK2)))

ct <- read_tsv(file.path(dir, "contigs.tsv"), show_col_types = FALSE) |>
  filter(status %in% c("kept", "contaminant"), within + outside >= minc) |>
  mutate(decision = factor(ifelse(status == "kept", "Kept in its bin", "Flagged as contaminant"),
                           levels = c("Kept in its bin", "Flagged as contaminant")))

# ---- A: within vs outside ---------------------------------------------------------------------------
qa <- ct |> mutate(side = ifelse(within >= outside, "mostly within", "mostly outside")) |>
  count(decision, side) |> group_by(decision) |> mutate(lab = sprintf("%s: %s (%s)", side, comma(n), percent(n / sum(n), 1))) |>
  summarise(lab = paste(lab, collapse = "\n"), .groups = "drop")
lim <- c(1, max(c(ct$within, ct$outside)) + 1)
pA <- ggplot(ct, aes(within + 1, outside + 1)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = INK2, linewidth = 0.3) +
  geom_point(aes(colour = p_own), size = 1, alpha = 0.6) +
  geom_text(data = qa, aes(x = 1.2, y = lim[2], label = lab), hjust = 0, vjust = 1, size = 2.8, colour = INK, inherit.aes = FALSE) +
  scale_x_log10(breaks = 10^(0:6) + 1, labels = c(0, comma(10^(1:6))), limits = lim) +
  scale_y_log10(breaks = 10^(0:6) + 1, labels = c(0, comma(10^(1:6))), limits = lim) +
  scale_colour_gradient(low = ORANGE, high = BLUE, limits = c(0, 1), labels = percent, name = "epirefine\nP(own bin)") +
  facet_wrap(~decision) +
  labs(x = "Hi-C contacts with other contigs of its bin", y = "Hi-C contacts with contigs of other bins",
       title = "A  Hi-C contacts inside vs outside the bin, per contig",
       subtitle = sprintf("Binned contigs with >= %d contacts to binned contigs; above the dashed line most contacts leave the bin", minc))

# ---- B: within share vs epirefine probability ----------------------------------------------------------
edges <- c(0, 0.05, 0.5, 0.95, 0.99, 1.0001)
labs_p <- c("<0.05", "0.05-0.5", "0.5-0.95", "0.95-0.99", ">=0.99")
ctb <- ct |> filter(!is.na(p_own)) |> mutate(pband = factor(labs_p[findInterval(p_own, edges)], levels = labs_p))
nb <- ctb |> count(pband)
pB <- ggplot(ctb, aes(pband, within_share)) +
  geom_boxplot(fill = "#dbe8f8", colour = BLUE, outlier.size = 0.4, outlier.alpha = 0.4, width = 0.6) +
  geom_text(data = nb, aes(x = pband, y = 1.07, label = paste0("n = ", comma(n))), size = 2.7, colour = INK2, inherit.aes = FALSE) +
  scale_y_continuous(labels = percent, breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  coord_cartesian(ylim = c(0, 1.1)) +
  labs(x = "epirefine P(contig belongs to its bin)", y = "within-bin share of Hi-C contacts",
       title = "B  Within-bin Hi-C share by epirefine probability",
       subtitle = "within / (within + other bins); a reliable P(own bin) rises with the share")

# ---- C: per bin, refined vs original ------------------------------------------------------------------
bn <- read_tsv(file.path(dir, "bins.tsv"), show_col_types = FALSE, col_types = cols(bin = "c", set = "c"))
orig <- bn |> filter(set == "original") |> select(bin, orig = within_share, orig_n = within)
cmp <- bn |> filter(set != "original") |> inner_join(orig, by = "bin") |> filter(!is.na(orig), !is.na(within_share))
cc <- cmp |> mutate(d = within_share - orig) |>
  group_by(set) |> summarise(lab = sprintf("higher %d, lower %d, same %d", sum(d > 0.005), sum(d < -0.005), sum(abs(d) <= 0.005)), .groups = "drop")
pC <- ggplot(cmp, aes(orig, within_share)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = INK2, linewidth = 0.3) +
  geom_point(colour = BLUE, size = 1, alpha = 0.6) +
  geom_text(data = cc, aes(x = 0.02, y = 0.98, label = lab), hjust = 0, size = 2.8, colour = INK, inherit.aes = FALSE) +
  scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) + coord_equal() +
  facet_wrap(~set) +
  labs(x = "within-bin share, original bins", y = "within-bin share, refined bins",
       title = "C  Per bin: within-bin share of Hi-C contacts after refinement",
       subtitle = "above the diagonal: the refined bin keeps more of its contacts inside")

title <- if (nzchar(opt$sample)) paste0(opt$sample, ": ") else ""
fig <- (pA / (pB | pC)) + plot_layout(heights = c(1, 1)) +
  plot_annotation(title = paste0(title, "Hi-C contacts inside vs outside the bin"),
                  theme = theme(plot.title = element_text(face = "bold", size = 14)))
ggsave(file.path(dir, "within_vs_outside.pdf"), fig, width = 13, height = 11)
ggsave(file.path(dir, "within_vs_outside.png"), fig, width = 13, height = 11, dpi = 150, bg = "white")
cat("wrote", file.path(dir, "within_vs_outside.pdf"), "\n")
