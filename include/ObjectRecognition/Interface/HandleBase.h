//
// Created by zhangye on 2020/9/15.
//

#ifndef ORB_SLAM3_HANDLEBASE_H
#define ORB_SLAM3_HANDLEBASE_H
namespace ObjRecognitionInterface {
class HandleBase {
public:
    enum {
        HANDLE_UNINITIALIZED = 0,
        HANDLE_INITIALIZED,
        HANDLE_BASE_STATE_END
    };

    virtual ~HandleBase() = 0;

    int HandleState() const;

    bool IsInitializedState() const;
    bool IsUninitializedState() const;

protected:
    HandleBase();

    void ToInitializedState();
    void ToUninitializedState();

    void SetHandleState(int handleState);

private:
    int m_handleState;

    HandleBase(const HandleBase &) = delete;
    HandleBase &operator=(const HandleBase &) = delete;
};

inline HandleBase::HandleBase() : m_handleState(HANDLE_UNINITIALIZED) {
}

inline HandleBase::~HandleBase() {
}

inline bool HandleBase::IsInitializedState() const {
    return (HANDLE_INITIALIZED == HandleState());
}

inline void HandleBase::ToInitializedState() {
    SetHandleState(HANDLE_INITIALIZED);
}

inline bool HandleBase::IsUninitializedState() const {
    return (HANDLE_UNINITIALIZED == HandleState());
}

inline void HandleBase::ToUninitializedState() {
    SetHandleState(HANDLE_UNINITIALIZED);
}

inline int HandleBase::HandleState() const {
    return m_handleState;
}

inline void HandleBase::SetHandleState(int state) {
    m_handleState = state;
}
} // namespace ObjRecognitionInterface
#endif // ORB_SLAM3_HANDLEBASE_H
