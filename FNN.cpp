#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

/* class FNN {
public:
    FNN(int input_size, int hidden_size, int output_size, std::string weight_init = "random") :
        input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        if (weight_init == "he") {
            // Inicialización de pesos con He
            scale = std::sqrt(2.0 / input_size);
        }
        else if (weight_init == "glorot") {
            // Inicialización de pesos con Glorot
            scale = std::sqrt(2.0 / (input_size + output_size));
        }
        else if (weight_init == "random") {
            scale = 1.0;
        }
        else {
            throw std::invalid_argument("Tipo de inicializacion no reconocida.");
        }

        // Inicializar pesos
        W1 = Eigen::MatrixXd::Random(input_size, hidden_size) * scale;
        b1 = Eigen::MatrixXd::Zero(1, hidden_size);
        W2 = Eigen::MatrixXd::Random(hidden_size, output_size) * scale;
        b2 = Eigen::MatrixXd::Zero(1, output_size);
    }

    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) {
        return 1.0 / (1.0 + (-x.array()).exp());
    }

    Eigen::MatrixXd softmax(const Eigen::MatrixXd& x) {
        Eigen::MatrixXd exps = (x.array() - x.maxCoeff()).exp();
        return exps.array() / exps.sum();
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X) {
        z1 = X * W1 + b1;
        a1 = sigmoid(z1);
        z2 = a1 * W2 + b2;
        a2 = softmax(z2);
        return a2;
    }

    void backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, double learning_rate = 0.01) {
        int m = X.rows();

        error = a2 - y;
        grad_W2 = a1.transpose() * error / m;
        grad_b2 = error.colwise().sum() / m;
        error2 = error * (W2.transpose().cast<double>().matrix()) * (a1.array().cast<double>().matrix()) * (1 - a1.array()).cast<double>().matrix();


        grad_W1 = X.transpose() * error2 / m;
        grad_b1 = error2.colwise().sum() / m;

        W2 -= learning_rate * grad_W2;
        b2 -= learning_rate * grad_b2;
        W1 -= learning_rate * grad_W1;
        b1 -= learning_rate * grad_b1;
    }

    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, int epochs = 10, double learning_rate = 0.01) {
        for (int i = 0; i < epochs; ++i) {
            std::cout << "Epoch #" << i << std::endl;
            Eigen::MatrixXd y_pred = forward(X);
            backward(X, y, learning_rate);
            std::cout << "Loss: " << loss_cross_entropy(y, y_pred) << std::endl;
        }
    }

    int argmax(const Eigen::MatrixXd& matrix) {
        int index = 0;
        double max_value = matrix(0, 0);

        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                if (matrix(i, j) > max_value) {
                    max_value = matrix(i, j);
                    index = i * matrix.cols() + j;
                }
            }
        }

        return index;
    }

    double evaluate(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
        Eigen::MatrixXd y_pred = forward(X);
        int correct_predictions = 0;

        for (int i = 0; i < X.rows(); i++) {
            int predicted_class = argmax(y_pred.row(i));
            int true_class = argmax(y.row(i));
            if (predicted_class == true_class) {
                correct_predictions++;
            }
        }

        return static_cast<double>(correct_predictions) / X.rows();
    }

    Eigen::MatrixXd predict(const Eigen::MatrixXd& X) {
        return forward(X).unaryExpr([](double x) { return x > 0.5 ? 1.0 : 0.0; });
    }

    void saveWeightsBiases() {
        W1.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ";\n", "[", "]", "", ""));
        std::cout << W1 << std::endl;
        W2.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ";\n", "[", "]", "", ""));
        std::cout << W2 << std::endl;
        b1.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ";\n", "[", "]", "", ""));
        std::cout << b1 << std::endl;
        b2.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ";\n", "[", "]", "", ""));
        std::cout << b2 << std::endl;
    }

    void loadWeights() {
        // Implementar la carga de pesos desde archivos
        // Puedes usar las funciones de Eigen para cargar matrices desde archivos
    }

    double loss_cross_entropy(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_pred) {
        return (-y.array() * (y_pred.array() + 1e-10).log()).mean();
    }

private:
    int input_size;
    int hidden_size;
    int output_size;
    double scale;
    Eigen::MatrixXd W1;
    Eigen::MatrixXd b1;
    Eigen::MatrixXd W2;
    Eigen::MatrixXd b2;
    Eigen::MatrixXd z1;
    Eigen::MatrixXd a1;
    Eigen::MatrixXd z2;
    Eigen::MatrixXd a2;
    Eigen::MatrixXd error;
    Eigen::MatrixXd grad_W2;
    Eigen::MatrixXd grad_b2;
    Eigen::MatrixXd error2;
    Eigen::MatrixXd grad_W1;
    Eigen::MatrixXd grad_b1;
};

Eigen::MatrixXd loadMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo: " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    file.close();

    int rows = data.size();
    int cols = (rows > 0) ? data[0].size() : 0;

    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

Eigen::MatrixXd oneHotEncode(const Eigen::VectorXi& labels, int num_classes) {
    Eigen::MatrixXd one_hot_matrix = Eigen::MatrixXd::Zero(labels.size(), num_classes);
    
    for (int i = 0; i < labels.size(); i++) {
        one_hot_matrix(i, labels(i)) = 1.0;
    }

    return one_hot_matrix;
} */

Eigen::MatrixXd loadMatrixFromCSV(const std::string& filename) {
    // Abrir el archivo CSV
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo: " << filename << std::endl;
        exit(1);
    }

    // Leer el archivo CSV línea por línea
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    // Cerrar el archivo CSV
    file.close();

    // Calcular el número de filas y columnas
    int rows = data.size();
    int cols = (rows > 0) ? data[0].size() : 0;

    // Crear una matriz Eigen con los datos
    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

int main() {
    // Cargar datos desde archivos CSV
    Eigen::MatrixXd X_train = loadMatrixFromCSV("X_train.csv");
    Eigen::MatrixXd X_test = loadMatrixFromCSV("X_test.csv");
    Eigen::MatrixXd y_train = loadMatrixFromCSV("y_train.csv").cast<int>();
    Eigen::MatrixXd y_test = loadMatrixFromCSV("y_test.csv").cast<int>();


    // Imprimir dimensiones de los datos
    std::cout << "Dimensiones de X_train: " << X_train.rows() << " x " << X_train.cols() << std::endl;
    std::cout << "Dimensiones de X_test: " << X_test.rows() << " x " << X_test.cols() << std::endl;
    std::cout << "Dimensiones de y_train: " << y_train.rows() << " x " << y_train.cols() << std::endl;
    std::cout << "Dimensiones de y_test: " << y_test.rows() << " x " << y_test.cols() << std::endl;


    /* // Codificar las etiquetas en formato one-hot
    int num_classes = 10;
    Eigen::MatrixXd y_train_one_hot = oneHotEncode(y_train, num_classes);
    Eigen::MatrixXd y_test_one_hot = oneHotEncode(y_test, num_classes);

    int input_size = X_train.cols();
    int hidden_size = 100;
    int epochs = 100;
    int output_size = num_classes;

    // Crear y entrenar el modelo
    FNN model(input_size, hidden_size, output_size, "random");
    model.train(X_train, y_train_one_hot, epochs);

    // Evaluar el modelo
    double accuracy = model.evaluate(X_test, y_test_one_hot);
    std::cout << "Accuracy: " << accuracy << std::endl;
    
    // Realizar predicciones
    for (int i = 50; i < 100; i++) {
        Eigen::MatrixXd predict = model.predict(X_test.row(i));
        int predicted_class = model.argmax(predict.row(0));
        int real_class = y_test(i);
        std::cout << "Clase predicha: " << predicted_class << std::endl;
        std::cout << "Clase real: " << real_class << std::endl;
        std::cout << "-----------" << std::endl;
    } */

    return 0;
}
