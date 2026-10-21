#!/usr/bin/env Rscript
# Compare bin sets (original and refined) by CheckM2 quality and by which contigs moved.
#
# Rscript compare_bins.R --sample ZymoOral \
#   --checkm2 original/quality_report.tsv nanomotif/quality_report.tsv pbinner/quality_report.tsv \
#   --contig-bin contig_bin.tsv nanomotif/new_contig_bin.tsv pbinner/new_contig_bin.tsv \
#   --methods original nanomotif pbinner --lengths contig_lengths.tsv --out comparison/
#
# The first method is the reference the others are compared with. Quality tiers follow the MIMAG
# completeness / contamination thresholds (rRNA and tRNA criteria are not checked):
#   high: completeness >= 90 and contamination < 5;  medium: >= 50 and < 10;  low: the rest.
# Writes summary.tsv (one row per method), bins.tsv (one row per bin and method) and
# bin_refinement.pdf / .png.

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(purrr); library(ggplot2); library(patchwork); library(scales)
})

# ---- arguments ---------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
opt <- list()
key <- NULL
for (a in args) {
  if (startsWith(a, "--")) { key <- sub("^--", "", a); opt[[key]] <- character(0) }
  else opt[[key]] <- c(opt[[key]], a)
}
need <- c("checkm2", "contig-bin", "methods", "out")
miss <- setdiff(need, names(opt))
if (length(miss)) stop("missing arguments: ", paste0("--", miss, collapse = " "))
methods <- opt$methods
if (length(opt$checkm2) != length(methods) || length(opt[["contig-bin"]]) != length(methods))
  stop("give one --checkm2 report and one --contig-bin table per method")
sample <- if (length(opt$sample)) opt$sample else "sample"
out <- opt$out
dir.create(out, showWarnings = FALSE, recursive = TRUE)
ref <- methods[1]
labels <- c(original = "Original bins", nanomotif = "Nanomotif", pbinner = "pbinner")
lab <- function(m) ifelse(m %in% names(labels), labels[m], m)
method_colours <- c("Original bins" = "#8C8C8C", "Nanomotif" = "#D55E00", "pbinner" = "#0072B2")

tier_of <- function(comp, cont) {
  case_when(is.na(comp) ~ "no bin",
            comp >= 90 & cont < 5 ~ "high",
            comp >= 50 & cont < 10 ~ "medium",
            TRUE ~ "low")
}
tier_levels <- c("high", "medium", "low", "no bin")

# ---- inputs ------------------------------------------------------------------------------------
read_report <- function(path, method) {
  read_tsv(path, show_col_types = FALSE) |>
    transmute(bin = Name, completeness = Completeness, contamination = Contamination,
              genome_size = Genome_Size, contigs = Total_Contigs, method = method)
}
read_contig_bin <- function(path, method) {
  x <- read_tsv(path, col_names = FALSE, show_col_types = FALSE, col_types = "cc")
  x <- x[, 1:2]
  names(x) <- c("contig", "bin")
  x |> filter(!(contig %in% c("contig", "contig_id", "seq_name"))) |> mutate(method = method)
}
q <- map2_dfr(opt$checkm2, methods, read_report)
cb <- map2_dfr(opt[["contig-bin"]], methods, read_contig_bin)
lengths <- if (length(opt$lengths)) read_tsv(opt$lengths, show_col_types = FALSE, col_types = "ci") else NULL

# every bin of the reference, under every method (a bin can lose all its contigs: "no bin")
bins <- expand_grid(bin = unique(q$bin[q$method == ref]), method = methods) |>
  left_join(q, by = c("bin", "method")) |>
  mutate(tier = factor(tier_of(completeness, contamination), levels = tier_levels))
base <- bins |> filter(method == ref) |>
  select(bin, completeness_ref = completeness, contamination_ref = contamination, tier_ref = tier)
bins <- bins |> left_join(base, by = "bin") |>
  mutate(d_completeness = completeness - completeness_ref,
         d_contamination = contamination - contamination_ref,
         change = case_when(
           method == ref ~ "reference",
           is.na(completeness) ~ "bin removed",
           d_contamination <= -0.5 & d_completeness >= -0.5 ~ "better",
           d_completeness >= 0.5 & d_contamination <= 0.5 ~ "better",
           d_contamination >= 0.5 | d_completeness <= -0.5 ~ "worse",
           TRUE ~ "unchanged"))

# contigs moved relative to the reference binning
ref_cb <- cb |> filter(method == ref) |> select(contig, ref_bin = bin)
moved <- map_dfr(setdiff(methods, ref), function(m) {
  new <- cb |> filter(method == m) |> select(contig, new_bin = bin)
  x <- full_join(ref_cb, new, by = "contig")
  tibble(method = m,
         contig = x$contig,
         ref_bin = x$ref_bin, new_bin = x$new_bin) |>
    mutate(kind = case_when(
      !is.na(ref_bin) & is.na(new_bin) ~ "removed",
      is.na(ref_bin) & !is.na(new_bin) ~ "added",
      !is.na(ref_bin) & !is.na(new_bin) & ref_bin != new_bin ~ "moved",
      TRUE ~ "kept"))
})
if (!is.null(lengths)) moved <- moved |> left_join(lengths, by = "contig") else moved$length <- NA_integer_

# ---- tables ------------------------------------------------------------------------------------
move_summary <- moved |> group_by(method) |>
  summarise(contigs_removed = sum(kind == "removed"), contigs_added = sum(kind == "added"),
            contigs_moved = sum(kind == "moved"),
            bp_removed = sum(length[kind %in% c("removed", "moved")], na.rm = TRUE),
            bp_added = sum(length[kind %in% c("added", "moved")], na.rm = TRUE), .groups = "drop")
summary <- bins |> group_by(method) |>
  summarise(bins = sum(!is.na(completeness)),
            high = sum(tier == "high"), medium = sum(tier == "medium"), low = sum(tier == "low"),
            median_completeness = median(completeness, na.rm = TRUE),
            median_contamination = median(contamination, na.rm = TRUE),
            mean_contamination = mean(contamination, na.rm = TRUE),
            bins_better = sum(change == "better"), bins_worse = sum(change == "worse"),
            bins_removed = sum(change == "bin removed"), .groups = "drop") |>
  left_join(move_summary, by = "method") |>
  mutate(sample = sample, method = factor(method, levels = methods)) |>
  arrange(method) |> relocate(sample)
write_tsv(summary, file.path(out, "summary.tsv"))
write_tsv(bins |> mutate(sample = sample) |> relocate(sample), file.path(out, "bins.tsv"))

# ---- figures -----------------------------------------------------------------------------------
theme_set(theme_minimal(base_size = 10) +
            theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 11),
                  plot.title.position = "plot", legend.position = "bottom"))
bins_l <- bins |> mutate(label = factor(lab(method), levels = lab(methods)))

tiers <- bins_l |> filter(tier != "no bin") |> count(label, tier) |>
  mutate(tier = factor(tier, levels = rev(c("high", "medium", "low"))))
p_tiers <- ggplot(tiers, aes(x = n, y = label, fill = tier)) +
  geom_col(width = 0.6, colour = "white", linewidth = 0.4) +
  geom_text(aes(label = n), position = position_stack(vjust = 0.5), size = 3,
            colour = ifelse(tiers$tier == "high", "white", "grey15")) +
  scale_fill_manual(values = c(high = "#08519C", medium = "#6BAED6", low = "#C6DBEF"),
                    breaks = c("high", "medium", "low"),
                    labels = c("high (>=90% complete, <5% contaminated)", "medium (>=50%, <10%)", "low")) +
  scale_y_discrete(limits = rev) +
  labs(title = "Bins per quality tier", x = "Bins", y = NULL, fill = NULL)

changed <- bins_l |> filter(method != ref, change %in% c("better", "worse"))
p_arrows <- ggplot(changed) +
  annotate("rect", xmin = 0, xmax = 5, ymin = 90, ymax = 100, fill = "#08519C", alpha = 0.06) +
  geom_segment(aes(x = contamination_ref, y = completeness_ref, xend = contamination, yend = completeness,
                   colour = change), arrow = arrow(length = unit(1.6, "mm")), linewidth = 0.45) +
  geom_point(aes(x = contamination_ref, y = completeness_ref), colour = "#8C8C8C", size = 0.9) +
  facet_wrap(~label) +
  scale_colour_manual(values = c(better = "#1B7837", worse = "#B2182B")) +
  scale_x_sqrt(breaks = c(0, 1, 5, 10, 25, 50, 100)) +
  labs(title = "Bins whose quality changed", subtitle = "Arrow from the original bin to the refined bin; shaded: high-quality region",
       x = "Contamination (%, square-root scale)", y = "Completeness (%)", colour = NULL)

delta <- bins_l |> filter(method != ref, change != "bin removed")
p_delta <- ggplot(delta, aes(x = d_contamination, y = d_completeness)) +
  geom_hline(yintercept = 0, colour = "grey70") + geom_vline(xintercept = 0, colour = "grey70") +
  geom_point(aes(colour = label), alpha = 0.7, size = 1.4) +
  scale_colour_manual(values = method_colours, guide = "none") +
  facet_wrap(~label) +
  labs(title = "Change per bin", subtitle = "Upper left is better: less contamination, more completeness",
       x = "Change in contamination (percentage points)", y = "Change in completeness (points)")

mv <- moved |> filter(kind != "kept") |>
  mutate(direction = ifelse(kind == "added", "added to a bin", ifelse(kind == "removed", "removed from a bin", "moved to another bin")),
         label = factor(lab(method), levels = lab(methods))) |>
  group_by(label, direction) |> summarise(mb = sum(length, na.rm = TRUE) / 1e6, contigs = n(), .groups = "drop")
p_moves <- ggplot(mv, aes(x = mb, y = label, fill = direction)) +
  geom_col(width = 0.6, position = position_dodge(width = 0.7)) +
  geom_text(aes(label = paste0(contigs, " contigs")), position = position_dodge(width = 0.7), hjust = -0.1, size = 2.8) +
  scale_fill_manual(values = c("removed from a bin" = "#B2182B", "added to a bin" = "#1B7837", "moved to another bin" = "#762A83")) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.35))) +
  scale_y_discrete(limits = rev) +
  labs(title = "Sequence moved by each method", x = "Mb", y = NULL, fill = NULL)

page <- (p_tiers | p_moves) / p_arrows / p_delta +
  plot_layout(heights = c(1, 1.4, 1.2)) +
  plot_annotation(title = paste0(sample, ": bin quality before and after refinement (CheckM2)"),
                  theme = theme(plot.title = element_text(face = "bold", size = 13)))
ggsave(file.path(out, "bin_refinement.pdf"), page, width = 11, height = 13)
ggsave(file.path(out, "bin_refinement.png"), page, width = 11, height = 13, dpi = 150, bg = "white")

# ---- text summary ------------------------------------------------------------------------------
cat(sprintf("\n%s: %d bins in the original set\n", sample, sum(bins$method == ref & !is.na(bins$completeness))))
for (i in seq_len(nrow(summary))) {
  r <- summary[i, ]
  cat(sprintf("  %-10s high %3d  medium %3d  low %3d  | median completeness %.1f, contamination %.2f | better %d, worse %d",
              as.character(r$method), r$high, r$medium, r$low, r$median_completeness, r$median_contamination,
              r$bins_better, r$bins_worse))
  if (!is.na(r$contigs_removed)) cat(sprintf(" | contigs removed %d, added %d, moved %d", r$contigs_removed, r$contigs_added, r$contigs_moved))
  cat("\n")
}
