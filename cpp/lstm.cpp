#include <torch/extension.h>
#include <bits/stdc++.h>
#include <pybind11/pybind11.h>

// std::vector<at::Tensor> lltm_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias,
//     torch::Tensor old_h,
//     torch::Tensor old_cell) 
// {
    
//     auto X = torch::cat({old_h, input}, /*dim=*/1);

//     auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
//     auto gates = gate_weights.chunk(3, /*dim=*/1);

//     auto input_gate = torch::sigmoid(gates[0]);
//     auto output_gate = torch::sigmoid(gates[1]);
//     auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

//     auto new_cell = old_cell + candidate_cell * input_gate;
//     auto new_h = torch::tanh(new_cell) * output_gate;

//     return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights};

// }


// std::vector<torch::Tensor> lltm_backward(*state
//     torch::Tensor grad_h,*state
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights) 
// {
    
//     auto d_output_gate = torch::tanh(new_cell) * grad_h;
//     auto d_tanh_new_cell = output_gate * grad_h;
//     auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

//     auto d_old_cell = d_new_cell;
//     auto d_candidate_cell = input_gate * d_new_cell;
//     auto d_input_gate = candidate_cell * d_new_cell;

//     auto gates = gate_weights.chunk(3, /*dim=*/1);
//     d_input_gate *= d_sigmoid(gates[0]);
//     d_output_gate *= d_sigmoid(gates[1]);
//     d_candidate_cell *= d_elu(gates[2]);

//     auto d_gates =
//         torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

//     auto d_weights = d_gates.t().mm(X);
//     auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

//     auto d_X = d_gates.mm(weights);
//     const auto state_size = grad_h.size(1);
//     auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
//     auto d_input = d_X.slice(/*dim=*/1, state_size);

//     return {d_old_h, d_input, d_weights, d_bias, d_old_cell};

// }

using namespace torch;
using namespace std;

vector <at::Tensor> lstm_forward(Tensor input, Tensor weights, Tensor Wy, Tensor bias, Tensor by, Tensor ht_1, Tensor Ct_1)
{
    auto X = cat({ht_1, input}, 1);
    auto gate_weights = addmm(bias, X, weights.transpose(0,1));
    auto gates = gate_weights.chunk(4, 1);
    
    auto hf = sigmoid(gates[0]);
    auto hi = sigmoid(gates[1]);
    auto ho = sigmoid(gates[2]);
    auto hc = tanh(gates[3]);
    

    auto c = hf * Ct_1 + hi * hc;
    auto h = ho * tanh(c);

    auto y = addmm(by, h, Wy.transpose(0,1));

    return {h, c, y, Ct_1, hf, hi, ho, hc, X, gate_weights};
}

vector <at::Tensor> lstm_backward(Tensor dh, Tensor dc, Tensor h, Tensor c, Tensor y, Tensor Ct_1, Tensor hf, Tensor hi, Tensor ho, Tensor hc, Tensor X, Tensor gate_weights, Tensor weights)
{
    auto dy = softmax(y, 1);

    auto dWy = mm(h.transpose(0,1), dy);
    auto dby = dy;

    auto dho = tanh(c) * dh * ho * (1-ho);
    auto dhc = (ho * dh * (1 - c*c)) + dc;
    auto dhf = Ct_1 * dc;
    dhf *= hf * (1-hf);
    auto dhi = hc * dc;
    dhi *= hi * (1-hi);
    
    auto W = weights.chunk(4,0);

    auto dWf = mm(X.transpose(0,1), dhf);
    auto dbf = dhf;
    auto dXf = mm(dhf, W[0]);

    auto dWi = mm(X.transpose(0,1), dhi);
    auto dbi = dhi;
    auto dXi = mm(dhi, W[1]);

    auto dWo = mm(X.transpose(0,1), dho);
    auto dbo = dho;
    auto dXo = mm(dho, W[2]);

    auto dWc = mm(X.transpose(0,1), dhc);
    auto dbc = dhc;
    auto dXc = mm(dhc, W[3]);

    auto dX = dXf + dXi + dXo + dXc;
    const auto state_size = dh.size(1);
    dh = dX.slice(1, 0, state_size);
    auto d_input = dX.slice(1, state_size);

    auto d_weights = cat({dWf, dWi, dWo, dWc}, 1);
    auto d_bias = cat({dbf, dbi, dbo, dbc}, 1);

    dc = hf * dc; 
    return {d_input, d_weights.transpose(0,1), dWy.transpose(0,1), d_bias, dby, dh, dc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("forward", &lstm_forward, "LLTM forward");
    m.def("backward", &lstm_backward, "LLTM backward");
}