import SwiftUI
import CoreMotion

struct ContentView: View {
    @StateObject private var motionManager = MotionManager()
    
    var body: some View {
        VStack {
            Text("X: \(motionManager.x)")
            Text("Y: \(motionManager.y)")
            Text("Z: \(motionManager.z)")
            Text("Pitch: \(motionManager.pitch)")
            Text("Roll: \(motionManager.roll)")
            Text("Yaw: \(motionManager.yaw)")
        }
        .onAppear {
            motionManager.startUpdates()
        }
        .onDisappear {
            motionManager.stopUpdates()
        }
    }
}

class MotionManager: ObservableObject {
    private var motionManager: CMMotionManager
    private var timer: Timer?
    
    @Published var x: Double = 0.0
    @Published var y: Double = 0.0
    @Published var z: Double = 0.0
    @Published var pitch: Double = 0.0
    @Published var roll: Double = 0.0
    @Published var yaw: Double = 0.0
    
    init() {
        motionManager = CMMotionManager()
    }
    
    func startUpdates() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 0.1
            motionManager.startDeviceMotionUpdates()
            
            timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                if let data = self.motionManager.deviceMotion {
                    DispatchQueue.main.async {
                        self.x = data.userAcceleration.x
                        self.y = data.userAcceleration.y
                        self.z = data.userAcceleration.z
                        self.pitch = data.attitude.pitch
                        self.roll = data.attitude.roll
                        self.yaw = data.attitude.yaw
                        
                        self.sendDataToServer(x: self.x, y: self.y, z: self.z, pitch: self.pitch, roll: self.roll, yaw: self.yaw)
                    }
                }
            }
        }
    }
    
    func stopUpdates() {
        motionManager.stopDeviceMotionUpdates()
        timer?.invalidate()
    }
    
    func sendDataToServer(x: Double, y: Double, z: Double, pitch: Double, roll: Double, yaw: Double) {
        guard let url = URL(string: "http://your_server_ip:your_port/") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let parameters: [String: Any] = [
            "x": x,
            "y": y,
            "z": z,
            "pitch": pitch,
            "roll": roll,
            "yaw": yaw
        ]
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: parameters)
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error: \(error)")
            } else {
                print("Data sent successfully")
            }
        }
        
        task.resume()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
