package org.cytoscape.sample.internal;

import org.cytoscape.application.CyApplicationManager;
import org.cytoscape.view.model.CyNetworkViewManager;
import org.cytoscape.work.AbstractTask;
import org.cytoscape.work.TaskMonitor;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class AddViewTask extends AbstractTask {
    private final CyApplicationManager applicationManager;
    private final CyNetworkViewManager networkViewManager;

    // Constructor nhận CyApplicationManager và CyNetworkViewManager
    public AddViewTask(CyApplicationManager applicationManager, CyNetworkViewManager networkViewManager) {
        this.applicationManager = applicationManager;
        this.networkViewManager = networkViewManager;
    }

    @Override
    public void run(TaskMonitor taskMonitor) {
        taskMonitor.setTitle("Running Add View");

        try {
            // Kết nối tới server Python
            URL url = new URL("http://127.0.0.1:5000/run_add_view");  // Đảm bảo URL này chính xác
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");

            // Đọc phản hồi từ server
            int responseCode = connection.getResponseCode();
            if (responseCode == 200) {
                BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                String inputLine;
                StringBuilder response = new StringBuilder();

                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();

                taskMonitor.showMessage(TaskMonitor.Level.INFO, "Add View completed successfully! Response: " + response.toString());
            } else {
                taskMonitor.showMessage(TaskMonitor.Level.ERROR, "Error running Add View. Server responded with code: " + responseCode);
            }
        } catch (IOException e) {
            taskMonitor.showMessage(TaskMonitor.Level.ERROR, "Failed to connect to the server: " + e.getMessage());
        }
    }
}
