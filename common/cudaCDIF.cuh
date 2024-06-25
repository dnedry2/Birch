template <typename T>
class ProcessorCUDA : public Processor<T> {
public:
    volatile bool Available() override {
        return false;
    }
    void Process(Chunk<T>* input) override {

    }

    ProcessorCUDA(unsigned fftSize, FIRKernel* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, double sampRate, Complex<double>* tune, unsigned tuneLen) {

    }

private:
    volatile bool halt = false;
};