#!/usr/bin/env Rscript
# One point per contig: the probability methylation (epirefine) and Hi-C each give the same statement.
#
# Rscript plot_contig_probabilities.R --dir hic_probability/ [--sample ZymoFecal]
#
# Reads per_contig.tsv (hic/probability_vs_hic.py) and writes contig_probabilities.pdf / .png:
#   binned contigs (kept / flagged):  "belongs to its bin"   epirefine 1 - P(contaminant)  vs  Hi-C P(own bin)
#   included contigs:                 "belongs to bin X"     epirefine P(X)                vs  Hi-C P(X)
#   unbinned contigs:                 "belongs to some bin"  epirefine 1 - P(unknown)      vs  Hi-C 1 - P(no bin)
# Points are coloured by the contig's Hi-C contacts (more contacts = firmer Hi-C probability); the
# counts in each corner use 0.5 as the line between "yes" and "no".

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2); library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
opt <- list(sample = "")
key <- NULL
for (a in args) {
  if (startsWith(a, "--")) { key <- sub("^--", "", a); opt[[key]] <- character(0) }
  else opt[[key]] <- c(opt[[key]], a)
}
dir <- opt$dir
INK <- "#0b0b0b"; INK2 <- "#52514e"

d <- read_tsv(file.path(dir, "per_contig.tsv"), show_col_types = FALSE, col_types = cols(.default = "c")) |>
  mutate(across(c(p_meth, p_hic, hic_links), as.numeric),
         panel = case_when(status == "kept" ~ "Kept in its bin\nstatement: belongs to its bin",
                           status == "contaminant" ~ "Flagged as contaminant\nstatement: belongs to its bin",
                           status == "included" ~ "Included into a bin\nstatement: belongs to that bin",
                           TRUE ~ "Left unbinned\nstatement: belongs to some bin"),
         panel = factor(panel, levels = c("Kept in its bin\nstatement: belongs to its bin",
                                          "Flagged as contaminant\nstatement: belongs to its bin",
                                          "Left unbinned\nstatement: belongs to some bin",
                                          "Included into a bin\nstatement: belongs to that bin")))
counts <- d |>
  mutate(qx = p_meth >= 0.5, qy = p_hic >= 0.5) |>
  count(panel, qx, qy) |>
  group_by(panel) |> mutate(frac = n / sum(n)) |> ungroup() |>
  mutate(x = ifelse(qx, 0.97, 0.03), y = ifelse(qy, 1.07, -0.07), hj = ifelse(qx, 1, 0),
         lab = sprintf("%s (%s)", comma(n), percent(frac, accuracy = 1)))
ns <- d |> count(panel) |> mutate(lab = paste0("n = ", comma(n)))
# rank so the best-measured contigs are drawn last (on top)
d <- d |> arrange(hic_links)

p <- ggplot(d, aes(p_meth, p_hic)) +
  annotate("rect", xmin = -0.02, xmax = 0.5, ymin = -0.02, ymax = 0.5, fill = "#f4f4f2") +
  annotate("rect", xmin = 0.5, xmax = 1.02, ymin = 0.5, ymax = 1.02, fill = "#f4f4f2") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = INK2, linewidth = 0.3) +
  geom_point(aes(colour = hic_links), size = 1.1, alpha = 0.55,
             position = position_jitter(width = 0.008, height = 0.008, seed = 1)) +
  geom_text(data = counts, aes(x = x, y = y, label = lab, hjust = hj), size = 2.8, colour = INK, inherit.aes = FALSE) +
  geom_text(data = ns, aes(x = 0.5, y = 1.17, label = lab), size = 3, colour = INK2, inherit.aes = FALSE) +
  scale_colour_gradient(low = "#9fc2ee", high = "#0d3b78", trans = "log10", name = "Hi-C contacts\nto binned contigs") +
  scale_x_continuous(labels = percent, breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  scale_y_continuous(labels = percent, breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  coord_cartesian(xlim = c(-0.02, 1.02), ylim = c(-0.12, 1.2), clip = "off") +
  facet_wrap(~panel, nrow = 1) +
  labs(x = "Methylation (epirefine) probability", y = "Hi-C contact-map probability",
       title = paste0(if (nzchar(opt$sample)) paste0(opt$sample, ": ") else "",
                      "methylation vs Hi-C probability, one point per contig"),
       subtitle = paste("Each panel compares the probability both methods give the same statement about the contig.",
                        "Shaded corners: the methods agree (both >= 50% or both < 50%). Contigs with >= 10 Hi-C contacts; dashed: equal probability.",
                        sep = "\n")) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#e6e5e0", linewidth = 0.3),
        plot.title = element_text(face = "bold", colour = INK, size = 13), plot.subtitle = element_text(colour = INK2),
        strip.text = element_text(face = "bold", colour = INK), axis.text = element_text(colour = INK2),
        panel.spacing = unit(1.2, "lines"), legend.position = "right")
ggsave(file.path(dir, "contig_probabilities.pdf"), p, width = 15, height = 5.2)
ggsave(file.path(dir, "contig_probabilities.png"), p, width = 15, height = 5.2, dpi = 160, bg = "white")
write_tsv(counts |> select(panel, methylation_yes = qx, hic_yes = qy, contigs = n, share = frac),
          file.path(dir, "contig_probabilities_quadrants.tsv"))
cat("wrote", file.path(dir, "contig_probabilities.pdf"), "\n")
