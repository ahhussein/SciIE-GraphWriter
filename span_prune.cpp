#include <torch/extension.h>


#include <iostream>
#include <pybind11/pybind11.h>

using namespace std;


torch::Tensor extract_spans(
    torch::Tensor span_scores,
    torch::Tensor candidate_starts,
    torch::Tensor candidate_ends,
    torch::Tensor num_output_spans,
    int max_sentence_length,
    bool _sort_spans,
    bool _suppress_crossing
) {
    int num_sentences = span_scores.size(0);
    int num_input_spans = span_scores.size(1);
    int max_num_output_spans = 0;


    for (int i = 0; i < num_sentences; i++) {

      if (num_output_spans[i].item<int64_t>() > max_num_output_spans) {
        max_num_output_spans = num_output_spans[i].item<int64_t>();
      }
    }


    std::vector<std::vector<int>> sorted_input_span_indices(num_sentences,
                                                            std::vector<int>(num_input_spans));

    torch::Tensor output_span_indices = torch::ones({num_sentences, max_num_output_spans});

    for (int i = 0; i < num_sentences; i++) {
      std::iota(sorted_input_span_indices[i].begin(), sorted_input_span_indices[i].end(), 0);
      std::sort(sorted_input_span_indices[i].begin(), sorted_input_span_indices[i].end(),
                [&span_scores, &i](int j1, int j2) {
                  return span_scores[i][j2].item<int64_t>() < span_scores[i][j1].item<int64_t>();
                });
    }

    //for ( std::vector<int> &v : sorted_input_span_indices )
    //{
    //   for ( int x : v ) std::cout << x << ' ';
    //   std::cout << std::endl;
    //}

    for (int l = 0; l < num_sentences; l++) {
      std::vector<int> top_span_indices;
      std::unordered_map<int, int> end_to_earliest_start;
      std::unordered_map<int, int> start_to_latest_end;
      int current_span_index = 0, num_selected_spans = 0;
      while (num_selected_spans < num_output_spans[l].item<int64_t>() && current_span_index < num_input_spans) {
        int i = sorted_input_span_indices[l][current_span_index];
        bool any_crossing = false;
        if (_suppress_crossing) {
          const int& start = candidate_starts[l][i].item<int64_t>();
          const int& end = candidate_ends[l][i].item<int64_t>();

          for (int j = start; j <= end; ++j) {
            if (j > start) {
              auto latest_end_iter = start_to_latest_end.find(j);
              if (latest_end_iter != start_to_latest_end.end() && latest_end_iter->second > end) {
                // Given (), exists [], such that ( [ ) ]
                any_crossing = true;
                break;
              }
            }
            if (j < end) {
              auto earliest_start_iter = end_to_earliest_start.find(j);
              if (earliest_start_iter != end_to_earliest_start.end() && earliest_start_iter->second < start) {
                // Given (), exists [], such that [ ( ] )
                any_crossing = true;
                break;
              }
            }
          }
        }
        if (!any_crossing) {
          if (_sort_spans) {
            top_span_indices.push_back(i);
          } else {
            output_span_indices[l][num_selected_spans] = i;
          }
          ++num_selected_spans;
          if (_suppress_crossing) {
            // Update data struct.
            const int& start = candidate_starts[l][i].item<int64_t>();
            const int& end = candidate_ends[l][i].item<int64_t>();
            auto latest_end_iter = start_to_latest_end.find(start);
            if (latest_end_iter == start_to_latest_end.end() || end > latest_end_iter->second) {
              start_to_latest_end[start] = end;
            }
            auto earliest_start_iter = end_to_earliest_start.find(end);
            if (earliest_start_iter == end_to_earliest_start.end() || start < earliest_start_iter->second) {
              end_to_earliest_start[end] = start;
            }
          }
        }
        ++current_span_index;
      }
      // Sort and produce span indices.
      if (_sort_spans) {
        std::sort(top_span_indices.begin(), top_span_indices.end(),
                [&candidate_starts, &candidate_ends, &l] (int i1, int i2) {
                 if (i1 >= candidate_starts.size(1) || i1 < 0 || i2 >= candidate_starts.size(1) || i2 < 0) {
                    return false;
                 }
                  if (candidate_starts[l][i1].item<int64_t>() < candidate_starts[l][i2].item<int64_t>()) {
                    return true;
                  } else if (candidate_starts[l][i1].item<int64_t>() > candidate_starts[l][i1].item<int64_t>()) {
                    return false;
                  } else if (candidate_ends[l][i1].item<int64_t>() < candidate_ends[l][i1].item<int64_t>()) {
                    return true;
                  } else if (candidate_ends[l][i1].item<int64_t>() > candidate_ends[l][i1].item<int64_t>()) {
                    return false;
                  } else {
                    return i1 < i2;
                  }
                });


        for (int i = 0; i < num_output_spans[l].item<int64_t>(); ++i) {
          output_span_indices[l][i] = top_span_indices[i];
        }
      }

      // Pad with the last selected span index to ensure monotonicity.
      int last_selected = num_selected_spans - 1;
      if (last_selected >= 0) {
        for (int i = num_selected_spans; i < max_num_output_spans; ++i) {
          output_span_indices[l][i]= output_span_indices[l][last_selected].item<int64_t>();
        }
      }
    }

    return output_span_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extract_spans", &extract_spans, "extract_spans");
}