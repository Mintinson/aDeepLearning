#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <metann/data/matrix.hpp>
#include <metann/data/mnist.hpp>
#include <metann/data/scalar.hpp>
#include <metann/layers/compose/linear_layer.hpp>
#include <metann/layers/compose/single_layer.hpp>
#include <metann/layers/cost/negative_log_likelihood_layer.hpp>
#include <metann/layers/elementary/softmax_layer.hpp>
#include <metann/layers/fillers/constant_filler.hpp>
#include <metann/layers/fillers/var_scale_filler.hpp>
#include <metann/layers/initializer.hpp>
#include <metann/layers/interface_fun.hpp>
#include <metann/layers/optimizers/sgd_optimizer.hpp>
#include <metann/layers/policies/init_policy.hpp>
#include <metann/layers/policies/single_layer_policy.hpp>
#include <metann/layers/policies/update_policy.hpp>

using namespace metann;

namespace {
constexpr std::size_t kInputDim = 28 * 28;
constexpr std::size_t kClassCount = 10;

struct MnistWeightInitTag;
struct MnistBiasInitTag;

template <typename Element>
auto make_mnist_initializer(const std::uint32_t seed) {
    return make_initializer<Element, WeightInitializerIs<MnistWeightInitTag>, BiasInitializerIs<MnistBiasInitTag>>()
        .template setFiller<MnistWeightInitTag>(XavierFiller{seed})
        .template setFiller<MnistBiasInitTag>(ConstantFiller{0.0});
}

template <typename MatrixLike>
std::size_t argmax_row0(const MatrixLike& rowVector) {
    std::size_t bestIdx = 0;
    auto bestVal = rowVector(0, 0);
    for (std::size_t i = 1; i < rowVector.colNum(); ++i) {
        if (rowVector(0, i) > bestVal) {
            bestVal = rowVector(0, i);
            bestIdx = i;
        }
    }
    return bestIdx;
}

template <typename Element>
Matrix<Element, CPU> make_input_row(const MnistDataset<Element>& dataset, const std::size_t sampleIdx) {
    Matrix<Element, CPU> input(1, dataset.imageSize());
    const Element* src = dataset.imageData(sampleIdx);
    for (std::size_t i = 0; i < dataset.imageSize(); ++i) {
        input.setValue(0, i, src[i]);
    }
    return input;
}

template <std::size_t InputDim, std::size_t HiddenDim, std::size_t ClassCount, typename Element = float>
class MnistMlp {
public:
    using TrainHiddenLayer = InjectPolicy_t<SingleLayer, TanhAction, UpdatePolicy, FeedbackOutputPolicy>;
    using TrainOutputLayer = InjectPolicy_t<LinearLayer, UpdatePolicy, FeedbackOutputPolicy>;
    using TrainSoftmaxLayer = InjectPolicy_t<SoftmaxLayer, FeedbackOutputPolicy>;
    using TrainLossLayer = InjectPolicy_t<NegativeLogLikelihoodLayer, FeedbackOutputPolicy>;

    using EvalHiddenLayer = InjectPolicy_t<SingleLayer, TanhAction>;
    using EvalOutputLayer = InjectPolicy_t<LinearLayer>;
    using EvalSoftmaxLayer = InjectPolicy_t<SoftmaxLayer>;
    using BatchMatrix = Batch<Element, CPU, CategoryTags::Matrix>;

    explicit MnistMlp(const std::uint32_t seed)
        : m_hiddenTrain("mnist-hidden", InputDim, HiddenDim)
        , m_outputTrain("mnist-output", HiddenDim, ClassCount)
        , m_hiddenEval("mnist-hidden", InputDim, HiddenDim)
        , m_outputEval("mnist-output", HiddenDim, ClassCount)
        , m_initializer(make_mnist_initializer<Element>(seed)) {}

    void initialize() {
        layer_init(m_hiddenTrain, m_initializer, m_params);
        layer_init(m_outputTrain, m_initializer, m_params);
        syncEvalNetwork();
    }

    Element trainMiniBatch(const MnistDataset<Element>& dataset,
                           const std::vector<std::size_t>& order,
                           const std::size_t begin,
                           const std::size_t batchSize,
                           const Element learningRate) {
        if (begin >= order.size()) {
            throw std::runtime_error("trainMiniBatch: begin index out of range.");
        }

        const auto inputBatch = make_mnist_image_batch(dataset, order, begin, batchSize);
        const auto targetBatch = make_mnist_one_hot_batch(dataset, order, begin, batchSize, ClassCount);
        const std::size_t actualBatch = inputBatch.batchNum();

        if (actualBatch == 0) {
            throw std::runtime_error("trainMiniBatch: empty batch.");
        }

        Element lossSum = 0;

        for (std::size_t i = 0; i < actualBatch; ++i) {
            lossSum += forward_backward_single(inputBatch[i], targetBatch[i]);
        }

        const Element scaledLr = learningRate / static_cast<Element>(actualBatch);
        optim::layer_sgd_step<TrainHiddenLayer, Element, CPU>(m_hiddenTrain, m_initializer, m_params, scaledLr);
        optim::layer_sgd_step<TrainOutputLayer, Element, CPU>(m_outputTrain, m_initializer, m_params, scaledLr);

        return lossSum / static_cast<Element>(actualBatch);
    }

    void syncEvalNetwork() {
        layer_init(m_hiddenEval, m_initializer, m_params);
        layer_init(m_outputEval, m_initializer, m_params);
    }

    void saveModel(const std::string& filePath) const {
        std::ofstream out(filePath, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open model output file: " + filePath);
        }

        std::map<std::string, Matrix<Element, CPU>> savedParams;
        layer_save_weights(m_hiddenTrain, savedParams);
        layer_save_weights(m_outputTrain, savedParams);

        out << "metann_mnist_mlp_v1\n";
        out << "input_dim " << InputDim << "\n";
        out << "hidden_dim " << HiddenDim << "\n";
        out << "class_count " << ClassCount << "\n";
        out << "param_count " << savedParams.size() << "\n";

        out << std::setprecision(10);
        for (const auto& item : savedParams) {
            const auto& name = item.first;
            const auto& mat = item.second;

            out << "param " << name << ' ' << mat.rowNum() << ' ' << mat.colNum() << "\n";
            for (std::size_t r = 0; r < mat.rowNum(); ++r) {
                for (std::size_t c = 0; c < mat.colNum(); ++c) {
                    if (c > 0) {
                        out << ' ';
                    }
                    out << mat(r, c);
                }
                out << "\n";
            }
        }
    }

    std::size_t predictLabel(const Matrix<Element, CPU>& input) {
        auto hiddenPack = m_hiddenEval.feedForward(LayerIO::create().template set<LayerIO>(input));
        auto outputPack = m_outputEval.feedForward(hiddenPack);
        auto probPack = m_softmaxEval.feedForward(outputPack);

        const auto probs = evaluate(probPack.template get<LayerIO>());
        return argmax_row0(probs);
    }

private:
    Element forward_backward_single(const Matrix<Element, CPU>& input, const Matrix<Element, CPU>& target) {
        auto hiddenPack = m_hiddenTrain.feedForward(LayerIO::create().template set<LayerIO>(input));
        auto outputPack = m_outputTrain.feedForward(hiddenPack);
        auto probPack = m_softmaxTrain.feedForward(outputPack);

        auto lossPack = m_lossTrain.feedForward(
            CostLayerIO::create().template set<CostLayerIO>(probPack.template get<LayerIO>()).template set<CostLayerLabel>(
                target));
        const auto lossValue = evaluate(lossPack.template get<LayerIO>());

        auto gradToSoftmax =
            m_lossTrain.feedBackward(LayerIO::create().template set<LayerIO>(Scalar<Element, CPU>{static_cast<Element>(1)}));
        auto gradToOutput =
            m_softmaxTrain.feedBackward(LayerIO::create().template set<LayerIO>(gradToSoftmax.template get<CostLayerIO>()));
        auto gradToHidden = m_outputTrain.feedBackward(gradToOutput);
        m_hiddenTrain.feedBackward(gradToHidden);

        return lossValue.value();
    }

private:
    TrainHiddenLayer m_hiddenTrain;
    TrainOutputLayer m_outputTrain;
    TrainSoftmaxLayer m_softmaxTrain;
    TrainLossLayer m_lossTrain;

    EvalHiddenLayer m_hiddenEval;
    EvalOutputLayer m_outputEval;
    EvalSoftmaxLayer m_softmaxEval;

    decltype(make_mnist_initializer<Element>(0U)) m_initializer;
    std::map<std::string, Matrix<Element, CPU>> m_params;
};

struct RunConfig {
    std::string dataRoot = "data/mnist";
    std::string modelOut = "out/models/mnist_mlp_model.txt";
    std::size_t epochs = 5;
    std::size_t batchSize = 32;
    float learningRate = 0.03f;
    std::size_t trainLimit = 12000;
    std::size_t testLimit = 2000;
    std::size_t logEvery = 80;
    std::uint32_t seed = 42;
};

void print_usage() {
    std::cout << "MNIST training example (MetaNN)\n"
              << "Options:\n"
              << "  --data-root <path>      Directory with MNIST IDX files\n"
              << "  --model-out <path>      Output model file path\n"
              << "  --epochs <int>          Number of epochs (default: 5)\n"
              << "  --batch-size <int>      Samples per SGD update (default: 32)\n"
              << "  --learning-rate <float> SGD learning rate (default: 0.03)\n"
              << "  --train-limit <int>     Max train samples, 0 means full set\n"
              << "  --test-limit <int>      Max test samples, 0 means full set\n"
              << "  --log-every <int>       Print every N training updates\n"
              << "  --seed <int>            Random seed\n"
              << "  --help                  Show this help\n\n"
              << "Expected files under --data-root:\n"
              << "  train-images-idx3-ubyte\n"
              << "  train-labels-idx1-ubyte\n"
              << "  t10k-images-idx3-ubyte\n"
              << "  t10k-labels-idx1-ubyte\n";
}

RunConfig parse_args(const int argc, char** argv) {
    RunConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--help") {
            print_usage();
            std::exit(0);
        }
        if (arg == "--data-root") {
            cfg.dataRoot = require_value(arg);
            continue;
        }
        if (arg == "--model-out") {
            cfg.modelOut = require_value(arg);
            continue;
        }
        if (arg == "--epochs") {
            cfg.epochs = static_cast<std::size_t>(std::stoul(require_value(arg)));
            continue;
        }
        if (arg == "--batch-size") {
            cfg.batchSize = static_cast<std::size_t>(std::stoul(require_value(arg)));
            continue;
        }
        if (arg == "--learning-rate") {
            cfg.learningRate = std::stof(require_value(arg));
            continue;
        }
        if (arg == "--train-limit") {
            cfg.trainLimit = static_cast<std::size_t>(std::stoul(require_value(arg)));
            continue;
        }
        if (arg == "--test-limit") {
            cfg.testLimit = static_cast<std::size_t>(std::stoul(require_value(arg)));
            continue;
        }
        if (arg == "--log-every") {
            cfg.logEvery = static_cast<std::size_t>(std::stoul(require_value(arg)));
            continue;
        }
        if (arg == "--seed") {
            cfg.seed = static_cast<std::uint32_t>(std::stoul(require_value(arg)));
            continue;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    if (cfg.batchSize == 0) {
        throw std::runtime_error("batch-size must be greater than 0.");
    }
    if (cfg.learningRate <= 0.0f) {
        throw std::runtime_error("learning-rate must be greater than 0.");
    }

    return cfg;
}

template <typename ModelType, typename Element>
float evaluate_accuracy(ModelType& model, const MnistDataset<Element>& dataset, const std::size_t classCount) {
    model.syncEvalNetwork();

    std::size_t correct = 0;
    for (std::size_t i = 0; i < dataset.sampleCount(); ++i) {
        const auto input = make_input_row(dataset, i);
        const auto pred = model.predictLabel(input);
        const auto label = static_cast<std::size_t>(dataset.labels[i]);
        if (label < classCount && pred == label) {
            ++correct;
        }
    }

    if (dataset.sampleCount() == 0) {
        return 0.0f;
    }
    return static_cast<float>(correct) / static_cast<float>(dataset.sampleCount());
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const RunConfig cfg = parse_args(argc, argv);

        const std::filesystem::path root(cfg.dataRoot);
        const auto trainImagePath = (root / "train-images-idx3-ubyte").string();
        const auto trainLabelPath = (root / "train-labels-idx1-ubyte").string();
        const auto testImagePath = (root / "t10k-images-idx3-ubyte").string();
        const auto testLabelPath = (root / "t10k-labels-idx1-ubyte").string();

        std::cout << "Loading MNIST from: " << root.string() << '\n';
        auto trainSet = load_mnist_dataset<float>(trainImagePath, trainLabelPath, cfg.trainLimit, true);
        auto testSet = load_mnist_dataset<float>(testImagePath, testLabelPath, cfg.testLimit, true);

        if (trainSet.imageSize() != kInputDim) {
            throw std::runtime_error("Unexpected MNIST image shape. This example expects 28x28 images.");
        }

        std::cout << "Train samples=" << trainSet.sampleCount() << "  Test samples=" << testSet.sampleCount() << '\n';
        std::cout << "Config: epochs=" << cfg.epochs << " batch_size=" << cfg.batchSize
                  << " learning_rate=" << cfg.learningRate << "\n\n";

        using Model = MnistMlp<kInputDim, 128, kClassCount, float>;
        Model model(cfg.seed);
        model.initialize();

        auto trainOrder = make_index_order(trainSet.sampleCount());
        std::mt19937 rng(cfg.seed);

        for (std::size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
            std::shuffle(trainOrder.begin(), trainOrder.end(), rng);

            float epochLossSum = 0.0f;
            std::size_t updateCount = 0;

            for (std::size_t offset = 0; offset < trainOrder.size(); offset += cfg.batchSize) {
                const float loss = model.trainMiniBatch(trainSet, trainOrder, offset, cfg.batchSize, cfg.learningRate);
                epochLossSum += loss;
                ++updateCount;

                if (cfg.logEvery > 0 && (updateCount % cfg.logEvery == 0)) {
                    const float avgLoss = epochLossSum / static_cast<float>(updateCount);
                    std::cout << "epoch=" << epoch << " update=" << std::setw(4) << updateCount
                              << "  train_nll=" << std::fixed << std::setprecision(6) << avgLoss << '\n';
                }
            }

            const float trainLoss = epochLossSum / static_cast<float>(updateCount);
            const float testAcc = evaluate_accuracy(model, testSet, kClassCount);

            std::cout << "epoch=" << epoch << '/' << cfg.epochs << "  train_nll=" << std::fixed
                      << std::setprecision(6) << trainLoss << "  test_acc=" << std::setprecision(4) << (testAcc * 100.0f)
                      << "%\n";
        }

        const float finalAcc = evaluate_accuracy(model, testSet, kClassCount);
        const std::filesystem::path modelPath(cfg.modelOut);
        if (modelPath.has_parent_path()) {
            std::filesystem::create_directories(modelPath.parent_path());
        }
        model.saveModel(modelPath.string());

        std::cout << "\nFinal test accuracy=" << std::fixed << std::setprecision(4) << (finalAcc * 100.0f) << "%\n";
        std::cout << "Saved model to: " << modelPath.string() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "MNIST example failed: " << ex.what() << '\n';
        return 1;
    }
}
