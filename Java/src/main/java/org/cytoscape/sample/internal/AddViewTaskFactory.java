package org.cytoscape.sample.internal;

import org.cytoscape.application.CyApplicationManager;
import org.cytoscape.view.model.CyNetworkViewManager;
import org.cytoscape.work.TaskFactory;
import org.cytoscape.work.TaskIterator;

public class AddViewTaskFactory implements TaskFactory {
    private final CyApplicationManager applicationManager;
    private final CyNetworkViewManager networkViewManager;

    // Constructor nhận CyApplicationManager và CyNetworkViewManager
    public AddViewTaskFactory(CyApplicationManager applicationManager, CyNetworkViewManager networkViewManager) {
        this.applicationManager = applicationManager;
        this.networkViewManager = networkViewManager;
    }

    @Override
    public TaskIterator createTaskIterator() {
        // Tạo và trả về TaskIterator với AddViewTask
        return new TaskIterator(new AddViewTask(applicationManager, networkViewManager));
    }

    @Override
    public boolean isReady() {
        // Kiểm tra xem TaskFactory có sẵn sàng để chạy hay không
        return true;
    }
}
