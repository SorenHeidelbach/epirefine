#!/usr/bin/env Rscript
# Per binned contig: Hi-C within/outside-bin contact ratio against pbinner's contamination probability.
#
# Rscript plot_ratio_vs_contamination.R --dir hic_within/ [--sample ZymoFecal] [--min-contacts 10]
# Reads contigs.tsv (hic/within_vs_outside.py). y = log2((within + 1) / (outside + 1)): > 0 most of the
# contig's contacts to binned contigs stay in its bin, < 0 most go to other bins. x = pbinner
# P(contaminant) = 1 - P(own bin). Writes ratio_vs_contamination.pdf / .png into --dir.

suppressPackageStartupMessages({ library(dplyr); library(readr); library(ggplot2); library(scales) })

args <- commandArgs(trailingOnly = TRUE)
opt <- list(sample = "", `min-contacts` = "10")
key <- NULL
for (a in args) {
  if (startsWith(a, "--")) { key <- sub("^--", "", a); opt[[key]] <- character(0) }
  else opt[[key]] <- c(opt[[key]], a)
}
dir <- opt$dir
minc <- as.numeric(opt$`min-contacts`)
INK <- "#0b0b0b"; INK2 <- "#52514e"; BLUE <- "#2a78d6"; ORANGE <- "#eb6834"

d <- read_tsv(file.path(dir, "contigs.tsv"), show_col_types = FALSE) |>
  filter(status %in% c("kept", "contaminant"), within + outside >= minc, !is.na(p_own)) |>
  mutate(p_contam = 1 - p_own, ratio = log2((within + 1) / (outside + 1)),
         decision = factor(ifelse(status == "kept", "kept in its bin", "flagged as contaminant"),
                           levels = c("kept in its bin", "flagged as contaminant")))
# medians and middle 50% per probability band, drawn at the band's mean probability
edges <- c(0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0001)
band <- d |> mutate(b = findInterval(p_contam, edges)) |> group_by(b) |>
  summarise(x = mean(p_contam), med = median(ratio), lo = quantile(ratio, 0.25), hi = quantile(ratio, 0.75),
            n = n(), out = mean(ratio < 0), .groups = "drop")
q <- d |> mutate(side = ifelse(ratio >= 0, "within", "outside")) |> count(decision, side) |>
  group_by(decision) |> mutate(f = n / sum(n)) |> ungroup()
lab <- sprintf("kept: %s contigs, %s mostly outside\nflagged: %s contigs, %s mostly outside",
               comma(sum(q$n[q$decision == "kept in its bin"])), percent(q$f[q$decision == "kept in its bin" & q$side == "outside"], 1),
               comma(sum(q$n[q$decision == "flagged as contaminant"])), percent(q$f[q$decision == "flagged as contaminant" & q$side == "outside"], 1))
rho <- cor(d$p_contam, d$ratio, method = "spearman")

p <- ggplot(d, aes(p_contam, ratio)) +
  geom_hline(yintercept = 0, colour = INK2, linewidth = 0.4) +
  geom_vline(xintercept = 0.5, colour = INK2, linewidth = 0.3, linetype = "dashed") +
  geom_point(aes(colour = decision), size = 1.1, alpha = 0.5,
             position = position_jitter(width = 0.006, height = 0, seed = 1)) +
  geom_ribbon(data = band, aes(x = x, ymin = lo, ymax = hi), inherit.aes = FALSE, fill = INK, alpha = 0.12) +
  geom_line(data = band, aes(x = x, y = med), inherit.aes = FALSE, colour = INK, linewidth = 0.8) +
  geom_point(data = band, aes(x = x, y = med), inherit.aes = FALSE, colour = INK, size = 2) +
  annotate("text", x = 0.52, y = max(d$ratio), label = lab, hjust = 0, vjust = 1, size = 3, colour = INK) +
  annotate("text", x = 0.45, y = 0.4, label = "more contacts within the bin", hjust = 1, size = 2.8, colour = INK2) +
  annotate("text", x = 0.45, y = -0.4, label = "more contacts with other bins", hjust = 1, size = 2.8, colour = INK2) +
  scale_colour_manual(values = c(`kept in its bin` = BLUE, `flagged as contaminant` = ORANGE), name = "pbinner decision") +
  scale_x_continuous(labels = percent, breaks = seq(0, 1, 0.25)) +
  scale_y_continuous(breaks = seq(-16, 16, 2), labels = function(v) ifelse(v >= 0, paste0(2^v, ":1"), paste0("1:", 2^-v))) +
  labs(x = "pbinner probability that the contig is a contaminant (dashed: flag threshold 0.5)",
       y = "Hi-C contacts within its bin : with other bins (log scale)",
       title = paste0(if (nzchar(opt$sample)) paste0(opt$sample, ": ") else "",
                      "Hi-C within/outside-bin ratio vs pbinner contamination probability"),
       subtitle = sprintf("One point per binned contig with >= %d contacts to binned contigs; ratio of (within+1) to (outside+1).\nBlack: median and middle 50%% per probability band. Spearman rho = %.2f",
                          minc, rho)) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#e6e5e0", linewidth = 0.3),
        plot.title = element_text(face = "bold", colour = INK, size = 13), plot.subtitle = element_text(colour = INK2),
        axis.text = element_text(colour = INK2), legend.position = "top")
ggsave(file.path(dir, "ratio_vs_contamination.pdf"), p, width = 11, height = 7)
ggsave(file.path(dir, "ratio_vs_contamination.png"), p, width = 11, height = 7, dpi = 160, bg = "white")
write_tsv(band, file.path(dir, "ratio_vs_contamination_bands.tsv"))
cat(sprintf("Spearman %.3f\n", rho)); print(band)
