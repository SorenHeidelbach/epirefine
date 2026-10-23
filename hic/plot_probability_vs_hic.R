#!/usr/bin/env Rscript
# Plots for hic/probability_vs_hic.py: does pbinner's probability for a contig's bin match the
# confidence the Hi-C contact map gives that bin?
#
# Rscript plot_probability_vs_hic.R --dir hic_probability/ [--sample ZymoFecal]
#
# Reads contigs.tsv, calibration.tsv, correlation.tsv and params.tsv from --dir and writes
# probability_vs_hic.pdf / .png there:
#   A  pbinner probability vs Hi-C probability, contig counts on a probability grid, per comparison
#   B  calibration: per pbinner probability interval, the share of contigs Hi-C supports (P >= 0.5,
#      95% Wilson interval) and the mean Hi-C probability, against the diagonal
#   C  Hi-C verdict on pbinner's decisions: supports (Hi-C P >= 0.95), undecided, contradicts
#      (Hi-C P < 0.05) for the bin pbinner keeps the contig in, moves or includes it into, or flags
#      it out of

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(patchwork); library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
opt <- list(sample = "")
key <- NULL
for (a in args) {
  if (startsWith(a, "--")) { key <- sub("^--", "", a); opt[[key]] <- character(0) }
  else opt[[key]] <- c(opt[[key]], a)
}
dir <- opt$dir
contigs <- read_tsv(file.path(dir, "contigs.tsv"), show_col_types = FALSE, col_types = cols(.default = "c")) |>
  mutate(across(c(p, hic_p, hic_share, hic_links), as.numeric))
cal <- read_tsv(file.path(dir, "calibration.tsv"), show_col_types = FALSE)
corr <- read_tsv(file.path(dir, "correlation.tsv"), show_col_types = FALSE)
par <- read_tsv(file.path(dir, "params.tsv"), show_col_types = FALSE)
pv <- setNames(par$value, par$parameter)

BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; GREY <- "#b8b7b0"; INK <- "#0b0b0b"; INK2 <- "#52514e"
labels <- c(own = "Binned contig stays in its bin\n1 - P(contaminant)",
            alt = "Best other bin (move / include)\nP(best alternative)",
            map = "Most probable bin\nP(MAP)")
comp_lab <- function(x) factor(labels[x], levels = labels)
theme_set(theme_minimal(base_size = 10) +
          theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#e6e5e0", linewidth = 0.3),
                plot.title = element_text(face = "bold", colour = INK), plot.subtitle = element_text(colour = INK2),
                axis.text = element_text(colour = INK2), strip.text = element_text(face = "bold", colour = INK)))

# ---- A: probability grid -------------------------------------------------------------------------
edges <- c(0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001)
mids <- c("<0.05", "0.05-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-0.95", ">=0.95")
grid <- contigs |>
  mutate(px = factor(mids[findInterval(p, edges)], levels = mids),
         hy = factor(mids[findInterval(hic_p, edges)], levels = mids),
         comp = comp_lab(comparison)) |>
  count(comp, px, hy, .drop = FALSE) |>
  group_by(comp) |> mutate(frac = n / sum(n)) |> ungroup()
rho <- corr |> mutate(lab = sprintf("%s\nn = %s, Spearman rho = %.2f", labels[comparison], comma(contigs), spearman_p_vs_hic_p))
stripA <- setNames(rho$lab, labels[rho$comparison])
pA <- ggplot(grid, aes(px, hy)) +
  geom_tile(aes(fill = ifelse(n > 0, n, NA)), colour = "white", linewidth = 0.6) +
  geom_text(aes(label = ifelse(n > 0, comma(n), ""), colour = frac > 0.15), size = 2.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = INK2, linewidth = 0.3) +
  scale_fill_gradient(low = "#dbe8f8", high = "#0d3b78", trans = "log10", na.value = "#f4f4f2", name = "contigs") +
  scale_colour_manual(values = c(`FALSE` = INK, `TRUE` = "white"), guide = "none") +
  facet_wrap(~comp, nrow = 1, labeller = as_labeller(function(x) ifelse(x %in% names(stripA), stripA[x], x))) +
  labs(x = "pbinner probability for the bin", y = "Hi-C probability for the same bin",
       title = "A  pbinner probability against Hi-C contact-map probability",
       subtitle = sprintf("Contigs with >= %d contacts to binned contigs; Hi-C model: within-genome contact share f = %.2f, Dirichlet-multinomial alpha = %.0f, prior P(no bin) = %.1f",
                          as.integer(pv["min_links"]), pv["f"], pv["alpha"], pv["prior_none"])) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1), panel.grid.major = element_blank())

# ---- B: calibration ------------------------------------------------------------------------------
calp <- cal |> filter(contigs > 0) |> mutate(comp = comp_lab(comparison))
pB <- ggplot(calp, aes(mean_p)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = INK2, linewidth = 0.3) +
  geom_errorbar(aes(ymin = hic_supports_lo, ymax = hic_supports_hi), width = 0.02, colour = BLUE, linewidth = 0.5) +
  geom_line(aes(y = hic_supports), colour = BLUE, linewidth = 0.7) +
  geom_point(aes(y = hic_supports, size = contigs), colour = BLUE, fill = "white", shape = 21, stroke = 1) +
  geom_line(aes(y = mean_hic_p), colour = ORANGE, linewidth = 0.7, linetype = "22") +
  geom_point(aes(y = mean_hic_p), colour = ORANGE, size = 1.8) +
  scale_size_area(max_size = 6, breaks = c(10, 100, 1000), name = "contigs") +
  scale_x_continuous(limits = c(0, 1), labels = percent) + scale_y_continuous(limits = c(0, 1), labels = percent) +
  facet_wrap(~comp, nrow = 1) +
  labs(x = "mean pbinner probability (per probability interval)", y = NULL,
       title = "B  Calibration against Hi-C",
       subtitle = "Blue: share of contigs whose Hi-C probability for the bin is >= 0.5 (95% interval)   Orange: mean Hi-C probability   Dashed: perfect agreement")

# ---- C: Hi-C verdict per pbinner decision ----------------------------------------------------------
dec <- bind_rows(
  contigs |> filter(comparison == "own", status == "kept") |> mutate(decision = "kept in its bin"),
  contigs |> filter(comparison == "own", status %in% c("contaminant", "moved")) |> mutate(decision = "flagged as contaminant", hic_p = 1 - hic_p),
  contigs |> filter(comparison == "alt", status == "moved") |> mutate(decision = "moved to another bin"),
  contigs |> filter(comparison == "alt", status == "included") |> mutate(decision = "included into a bin"),
  contigs |> filter(comparison == "alt", status == "redundant") |> mutate(decision = "redundant (not included)"),
  contigs |> filter(comparison == "alt", status == "unbinned") |> mutate(decision = "left unbinned\n(best other bin)")
) |>
  mutate(verdict = case_when(hic_p >= 0.95 ~ "Hi-C supports", hic_p < 0.05 ~ "Hi-C contradicts", TRUE ~ "Hi-C undecided"),
         verdict = factor(verdict, levels = c("Hi-C contradicts", "Hi-C undecided", "Hi-C supports")),
         decision = factor(decision, levels = rev(c("kept in its bin", "flagged as contaminant", "moved to another bin",
                                                     "included into a bin", "redundant (not included)", "left unbinned\n(best other bin)"))))
decn <- dec |> count(decision, verdict) |> group_by(decision) |> mutate(frac = n / sum(n), total = sum(n)) |> ungroup()
pC <- ggplot(decn, aes(frac, decision, fill = verdict)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.5) +
  geom_text(aes(label = ifelse(frac >= 0.07, comma(n), "")), position = position_stack(vjust = 0.5), size = 2.7,
            colour = ifelse(decn$verdict == "Hi-C undecided", INK, "white")) +
  geom_text(data = distinct(decn, decision, total), aes(x = 1.01, y = decision, label = paste0("n = ", comma(total))),
            hjust = 0, size = 2.7, colour = INK2, inherit.aes = FALSE) +
  scale_fill_manual(values = c(`Hi-C contradicts` = ORANGE, `Hi-C undecided` = GREY, `Hi-C supports` = BLUE), name = NULL) +
  scale_x_continuous(labels = percent, expand = expansion(mult = c(0, 0.12))) +
  labs(x = "share of contigs", y = NULL, title = "C  Does Hi-C agree with pbinner's decisions?",
       subtitle = "Hi-C probability for the bin pbinner keeps the contig in, moves / includes it into, or (flagged) that it is NOT in its bin: >= 0.95 supports, < 0.05 contradicts") +
  theme(legend.position = "top", panel.grid.major.y = element_blank())

title <- if (length(opt$sample) && nzchar(opt$sample)) paste0(opt$sample, ": ") else ""
fig <- (pA / pB / pC) + plot_layout(heights = c(1.15, 0.9, 0.9)) +
  plot_annotation(title = paste0(title, "pbinner assignment probability vs Hi-C contact-map confidence"),
                  theme = theme(plot.title = element_text(face = "bold", size = 14)))
ggsave(file.path(dir, "probability_vs_hic.pdf"), fig, width = 13, height = 15)
ggsave(file.path(dir, "probability_vs_hic.png"), fig, width = 13, height = 15, dpi = 150, bg = "white")
dec |> count(decision, verdict) |> write_tsv(file.path(dir, "decision_verdicts.tsv"))
cat("wrote", file.path(dir, "probability_vs_hic.pdf"), "\n")
