
                def sim_update_v4(tstep):
                    ''' Function to get update position of agents.
                    '''
                    # Shuffle the order of the agents around
                    random.shuffle(self.S)
                    removal = []
                    # Record the density profile for this timestep in this list
                    this_tstep = [0]*fd.bins
                    for this_agent in self.S:
                        # Determine this agent's local density
                        this_density = self.K[this_agent]                        
                        # Determine the current velocity
                        this_velocity = self.V[this_agent]
                        # Determine the current location
                        this_location = self.L[this_agent]
                        # Determine the edge that an agent is on
                        this_edge = self.E[this_agent]
                        # Determine the length of the link
                        di = self.h.D[this_edge]
                        # Travelling from this node (ri)
                        ri = self.h.R[this_edge]                        
                        # To this node (ci)
                        ci = self.h.C[this_edge]                                                                                    
                        # If this is the timestep to take action
                        # - Determine the length of the link
                        # - Determine the amount of time it takes to reach the end
                        # - If the amount of time is more than one
                        #   - Move the agent location by the existing velocity
                        if self.A[this_agent] <= tstep:
                            # Time it takes to reach the end of the link at current velocity
                            time = (di-this_location)/this_velocity
                            # If there is more than one timestep to kill on this edge,
                            # determine when tstep when the next action needs to be taken.
                            # I can only see this being called in the first timestep.
                            if time > 1:
                                self.A[this_agent] = tstep + int(time)
                                # Move the agent along
                                this_location = this_location +  this_velocity
                            else:
                                # if tstep>100:
                                #     pdb.set_trace()
                                # This is how much time is left in this iteration
                                residual_time = 1 - time
                                # Keep going till the time in this iteration has been used up
                                while residual_time > 0:
                                    # If agent has reached the destination node
                                    if ci == self.X[this_agent]:
                                        this_location = di
                                        # Agent has reached its destination, record time
                                        self.T[this_agent] = tstep + time
                                        # Subtract the agent from this edge
                                        self.N[this_edge] -= 1
                                        # Remove agent from our list of agents
                                        removal.append(this_agent)
                                        # Reset the residual time to 0
                                        residual_time = 0
                                    # Agent has not reached the destination node
                                    else:
                                        next_ri = ci
                                        next_ci = self.h.route[self.X[this_agent]][ci]
                                        next_edge = self.h.EM[next_ri][next_ci]
                                        # If the next edge is blocked, equal likelihood of picking all available routes
                                        if self.blocked[next_edge] and self.random_successor:
                                            # Get list of all successor nodes
                                            all_succ = self.h.G.succ[next_ri]
                                            # Pick a random successor node
                                            next_ci = all_succ.keys()[random.randrange(len(all_succ))]
                                            # Determine the edge 
                                            next_edge = self.h.EM[next_ri][next_ci]
                                        # If there is no space
                                        if self.N[next_edge]+1 > self.EC[next_edge]:
                                            # Wait at the end of this link
                                            this_location = di
                                            # Check again at the next timestep
                                            self.A[this_agent] = tstep + 1
                                        # If there is enough space for the agent in the next link, only then proceed
                                        else:
                                            # Assign new nodes
                                            ri = next_ri
                                            ci = next_ci
                                            # Subtract agent away from previous edge
                                            self.N[this_edge] -= 1
                                            # Add agent to the new edge
                                            self.N[next_edge] += 1
                                            # Update this edge
                                            this_edge = next_edge
                                            # Update this density
                                            this_density = self.density(this_edge)
                                            # Calculate new velocity
                                            this_velocity = fd.v_dict[this_density]
                                            # Determine the length of the link
                                            di = self.h.D[this_edge]
                                            # Traverse the new link in the residual time
                                            this_location = this_velocity * residual_time
                                            if this_location > di:
                                                # If the agent moves further than is allowed, recover the residual time                                                
                                                residual_time = (this_location - di) / this_velocity
                                                this_location = di
                                            else:
                                                # If the agent remains within the link
                                                time = (di-this_location)/this_velocity                                               
                                                # Update the timestep to perform action
                                                self.A[this_agent] = tstep + int(time) 
                                                # Update the timestep to perform action
                                                self.V[this_agent] = this_velocity
                                                # Update the agent's local density
                                                self.K[this_agent] = this_density
                                                # Reset residual time to 0 so that we can exit the while loop                                            
                                                residual_time = 0
                        else:
                            # This is the only action that an agent needs to perform
                            # if it is not the tstep to perform action
                            this_location = this_location +  this_velocity
                        # Determine new position of an agent
                        this_node = np.array(self.h.nodes[ri])
                        that_node = np.array(self.h.nodes[ci])
                        offset = (that_node-this_node) * this_location / di
                        self.P[this_agent,:] = this_node + offset
                        # Update array of unit vectors if the agent marker is quiver
                        if self.agent_marker == 'quiver':
                            magnitude = np.linalg.norm(offset)
                            if magnitude > 0:
                                unit = offset/magnitude
                            else:
                                unit = offset
                            self.U[this_agent,:] = unit
                        # Update to current agent location
                        self.L[this_agent] = this_location
                        # Update to current agent edge
                        self.E[this_agent] = this_edge
                        if this_agent in self.track_list:
                            self.tracked_agent[this_agent].append(self.dist2exit(this_agent))
                        # Update journey time based on velocity on this link
                        if self.use_JT:
                            self.JT[ci][ri]['weight']=di/this_velocity
                        # Get the bin id in which to add the agent
                        bin = fd.which_bin(this_density)
                        # This is the density profile for a given agent over all timestep
                        self.KP_agent[this_agent,bin] = self.KP_agent[this_agent,bin] + 1
                        # This is the density profile for all agents per time step
                        this_tstep[bin] = this_tstep[bin] + 1
                        # This is the density profile for all agents for all edges over all timestep
                        self.KP_edges[this_edge,bin] = self.KP_edges[this_edge,bin] + 1
                    # Append the agglomerate density profile to our list of density profiles per timestep
                    self.KP_tstep.append(this_tstep)
                    # Remove agents that have reached destination
                    for r in removal:
                        self.S.remove(r)     


                def sim_update_v3(tstep):
                    ''' Function to get update position of agents.
                    '''
                    # Shuffle the order of the agents around
                    random.shuffle(self.S)
                    removal = []
                    # Record the density profile for this timestep in this list
                    this_tstep = [0]*fd.bins
                    for this_agent in self.S:
                        # Determine this agent's local density
                        this_density = self.K[this_agent]                        
                        # Determine the current velocity
                        this_velocity = self.V[this_agent]
                        # Determine the current location
                        this_location = self.L[this_agent]
                        # Determine the edge that an agent is on
                        this_edge = self.E[this_agent]
                        # Determine the length of the link
                        di = self.h.D[this_edge]
                        # Travelling from this node (ri)
                        ri = self.h.R[this_edge]                        
                        # To this node (ci)
                        ci = self.h.C[this_edge]                                                                                    
                        # If this is the timestep to take action
                        # - Determine the length of the link
                        # - Determine the amount of time it takes to reach the end
                        # - If the amount of time is more than one
                        #   - Move the agent location by the existing velocity
                        if self.A[this_agent] <= tstep:
                            # Time it takes to reach the end of the link at current velocity
                            time = (di-this_location)/this_velocity
                            # If there is more than one timestep to kill on this edge,
                            # determine when tstep when the next action needs to be taken.
                            # I can only see this being called in the first timestep.
                            if time > 1:
                                self.A[this_agent] = tstep + int(time)
                                # Move the agent along
                                this_location = this_location +  this_velocity
                            else:
                                # if tstep>100:
                                #     pdb.set_trace()
                                # This is how much time is left in this iteration
                                residual_time = 1 - time
                                # Keep going till the time in this iteration has been used up
                                while residual_time > 0:
                                    # If agent has reached the destination node
                                    if ci == self.X[this_agent]:
                                        this_location = di
                                        # Agent has reached its destination, record time
                                        self.T[this_agent] = tstep + time
                                        # Subtract the agent from this edge
                                        self.N[this_edge] -= 1
                                        # Remove agent from our list of agents
                                        removal.append(this_agent)
                                        # Reset the residual time to 0
                                        residual_time = 0
                                    # Agent has not reached the destination node
                                    else:
                                        next_ri = ci
                                        next_ci = self.h.route[self.X[this_agent]][ci]
                                        next_edge = self.h.EM[next_ri][next_ci]
                                        # If the next edge is blocked, equal likelihood of picking all available routes
                                        if self.blocked[next_edge] and self.random_successor:
                                            # Get list of all successor nodes
                                            all_succ = self.h.G.succ[next_ri]
                                            # Pick a random successor node
                                            next_ci = all_succ.keys()[random.randrange(len(all_succ))]
                                            # Determine the edge 
                                            next_edge = self.h.EM[next_ri][next_ci]
                                        # If there is no space
                                        if self.N[next_edge]+1 > self.EC[next_edge]:
                                            # Wait at the end of this link
                                            this_location = di
                                            # Check again at the next timestep
                                            self.A[this_agent] = tstep + 1
                                        # If there is enough space for the agent in the next link, only then proceed
                                        else:
                                            # Assign new nodes
                                            ri = next_ri
                                            ci = next_ci
                                            # Subtract agent away from previous edge
                                            self.N[this_edge] -= 1
                                            # Add agent to the new edge
                                            self.N[next_edge] += 1
                                            # Update this edge
                                            this_edge = next_edge
                                            # Update this density
                                            this_density = self.density(this_edge)
                                            # Calculate new velocity
                                            this_velocity = fd.v_dict[this_density]
                                            # Determine the length of the link
                                            di = self.h.D[this_edge]
                                            # Traverse the new link in the residual time
                                            this_location = this_velocity * residual_time
                                            if this_location > di:
                                                # If the agent moves further than is allowed, recover the residual time                                                
                                                residual_time = (this_location - di) / this_velocity
                                                this_location = di
                                            else:
                                                # If the agent remains within the link
                                                time = (di-this_location)/this_velocity                                               
                                                # Update the timestep to perform action
                                                self.A[this_agent] = tstep + int(time) 
                                                # Update the timestep to perform action
                                                self.V[this_agent] = this_velocity
                                                # Update the agent's local density
                                                self.K[this_agent] = this_density
                                                # Reset residual time to 0 so that we can exit the while loop                                            
                                                residual_time = 0
                        else:
                            # This is the only action that an agent needs to perform
                            # if it is not the tstep to perform action
                            this_location = this_location +  this_velocity
                        # Determine new position of an agent
                        this_node = np.array(self.h.nodes[ri])
                        that_node = np.array(self.h.nodes[ci])
                        offset = (that_node-this_node) * this_location / di
                        self.P[this_agent,:] = this_node + offset
                        # Update array of unit vectors if the agent marker is quiver
                        if self.agent_marker == 'quiver':
                            magnitude = np.linalg.norm(offset)
                            if magnitude > 0:
                                unit = offset/magnitude
                            else:
                                unit = offset
                            self.U[this_agent,:] = unit
                        # Update to current agent location
                        self.L[this_agent] = this_location
                        # Update to current agent edge
                        self.E[this_agent] = this_edge
                        if this_agent in self.track_list:
                            self.tracked_agent[this_agent].append(self.dist2exit(this_agent))
                        # Update journey time based on velocity on this link
                        if self.use_JT:
                            self.JT[ci][ri]['weight']=di/this_velocity
                        # Get the bin id in which to add the agent
                        bin = fd.which_bin(this_density)
                        # This is the density profile for a given agent over all timestep
                        self.KP_agent[this_agent,bin] = self.KP_agent[this_agent,bin] + 1
                        # This is the density profile for all agents per time step
                        this_tstep[bin] = this_tstep[bin] + 1
                        # This is the density profile for all agents for all edges over all timestep
                        self.KP_edges[this_edge,bin] = self.KP_edges[this_edge,bin] + 1
                    # Append the agglomerate density profile to our list of density profiles per timestep
                    self.KP_tstep.append(this_tstep)
                    # Remove agents that have reached destination
                    for r in removal:
                        self.S.remove(r)                

                def sim_update_v2(tstep):
                    ''' Function to get update position of agents.
                    '''
                    # Shuffle the order of the agents around
                    random.shuffle(self.S)
                    removal = []
                    # Record the density profile for this timestep in this list
                    this_tstep = [0]*fd.bins
                    for this_agent in self.S:
                        this_edge = self.E[this_agent]
                        this_density = self.K[this_agent]
                        this_velocity = fd.v_dict[this_density]
                        # Move the agent
                        this_location = self.L[this_agent] + this_velocity
                        di = self.h.D[this_edge]
                        # Travelling from this node (ri)
                        ri = self.h.R[this_edge]
                        # To this node (ci)
                        ci = self.h.C[this_edge]
                        # Has agent gone beyond the length of the link?
                        residual_di = this_location - di
                        # If the agent travels beyond the length of the link      
                        while residual_di > 0:
                            # Determine the amount of time remaining from last run
                            residual_time = residual_di/this_velocity
                            # Agent has reached the destination node
                            if ci == self.X[this_agent]:
                                this_location = di
                                # Agent has reached its destination, record time
                                self.T[this_agent] = tstep + 1 - residual_time
                                # Subtract the agent from this edge
                                self.N[this_edge] -= 1
                                # Remove agent from our list of agents
                                removal.append(this_agent)
                            # Agent has not reached the destination node
                            else:
                                next_ri = ci
                                next_ci = self.h.route[self.X[this_agent]][ci]
                                next_edge = self.h.EM[next_ri][next_ci]
                                # If the next edge is blocked, equal likelihood of picking all available routes
                                if self.blocked[next_edge] and self.random_successor:
                                    # Get list of all successor nodes
                                    all_succ = self.h.G.succ[next_ri]
                                    # Pick a random successor node
                                    next_ci = all_succ.keys()[random.randrange(len(all_succ))]
                                    # Determine the edge 
                                    next_edge = self.h.EM[next_ri][next_ci]
                                # If there is no space, then just wait at the end of the link
                                if self.N[next_edge]+1 > self.EC[next_edge]:
                                    this_location = di
                                # If there is enough space for the agent in the next link, only then proceed
                                else:
                                    # Assign new nodes
                                    ri = next_ri
                                    ci = next_ci
                                    # Subtract agent away from previous edge
                                    self.N[this_edge] -= 1
                                    # Add agent to the new edge
                                    self.N[next_edge] += 1
                                    # Update this edge
                                    this_edge = next_edge
                                    # Update this density
                                    this_density = self.density(this_edge)
                                    # Calculate new velocity
                                    this_velocity = fd.v_dict[this_density]
                                    # Update to current agent edge
                                    self.K[this_agent] = this_density
                                    # Assign the new distance taking into account the remaining time
                                    this_location = this_velocity * residual_time
                                    # Calculate new progress
                                    di = self.h.D[this_edge]
                            # Calculate the residual distance
                            residual_di = this_location - di
                        # Determine new position
                        this_node = np.array(self.h.nodes[ri])
                        that_node = np.array(self.h.nodes[ci])
                        pr = this_location / di
                        offset = (that_node-this_node) * pr
                        self.P[this_agent,:] = this_node + offset
                        # Update array of unit vectors if the agent marker is quiver
                        if self.agent_marker == 'quiver':
                            magnitude = np.linalg.norm(offset)
                            if magnitude > 0:
                                unit = offset/magnitude
                            else:
                                unit = offset
                            self.U[this_agent,:] = unit
                        # Update to current agent location
                        self.L[this_agent] = this_location
                        # Update to current agent edge
                        self.E[this_agent] = this_edge
                        if this_agent in self.track_list:
                            self.tracked_agent[this_agent].append(self.dist2exit(this_agent))
                        # Update journey time based on velocity on this link
                        if self.use_JT:
                            self.JT[ci][ri]['weight']=di/this_velocity
                        # Get the bin id in which to add the agent
                        bin = fd.which_bin(this_density)
                        # This is the density profile for a given agent over all timestep
                        self.KP_agent[this_agent,bin] = self.KP_agent[this_agent,bin] + 1
                        # This is the density profile for all agents per time step
                        this_tstep[bin] = this_tstep[bin] + 1
                        # This is the density profile for all agents for all edges over all timestep
                        self.KP_edges[this_edge,bin] = self.KP_edges[this_edge,bin] + 1
                    # Append the agglomerate density profile to our list of density profiles per timestep
                    self.KP_tstep.append(this_tstep)
                    # Remove agents that have reached destination
                    for r in removal:
                        self.S.remove(r)                        

                def sim_update_v1(tstep):
                    ''' Function to get update position of agents.
                    '''
                    # Shuffle the order of the agents around
                    random.shuffle(self.S)
                    removal = []
                    # Record the density profile for this timestep in this list
                    this_tstep = [0]*fd.bins
                    for this_agent in self.S:
                        this_edge = self.E[this_agent]
                        this_density = self.density(this_edge)
                        this_velocity = fd.v_dict[this_density]
                        # Move the agent
                        this_location = self.L[this_agent] + this_velocity
                        di = self.h.D[this_edge]
                        # Travelling from this node (ri)
                        ri = self.h.R[this_edge]
                        # To this node (ci)
                        ci = self.h.C[this_edge]
                        # Has agent gone beyond the length of the link?
                        residual_di = this_location - di
                        # If the agent travels beyond the length of the link      
                        while residual_di > 0:
                            # Determine the amount of time remaining from last run
                            residual_time = residual_di/this_velocity
                            # Agent has reached the destination node
                            if ci == self.X[this_agent]:
                                this_location = di
                                # Agent has reached its destination, record time
                                self.T[this_agent] = tstep + 1 - residual_time
                                # Subtract the agent from this edge
                                self.N[this_edge] -= 1
                                # Remove agent from our list of agents
                                removal.append(this_agent)
                            # Agent has not reached the destination node
                            else:
                                next_ri = ci
                                next_ci = self.h.route[self.X[this_agent]][ci]
                                next_edge = self.h.EM[next_ri][next_ci]
                                # If the next edge is blocked, equal likelihood of picking all available routes
                                if self.blocked[next_edge] and self.random_successor:
                                    # Get list of all successor nodes
                                    all_succ = self.h.G.succ[next_ri]
                                    # Pick a random successor node
                                    next_ci = all_succ.keys()[random.randrange(len(all_succ))]
                                    # Determine the edge 
                                    next_edge = self.h.EM[next_ri][next_ci]
                                # If there is no space, then just wait at the end of the link
                                if self.N[next_edge]+1 > self.EC[next_edge]:
                                    this_location = di
                                # If there is enough space for the agent in the next link, only then proceed
                                else:
                                    # Assign new nodes
                                    ri = next_ri
                                    ci = next_ci
                                    # Subtract agent away from previous edge
                                    self.N[this_edge] -= 1
                                    # Add agent to the new edge
                                    self.N[next_edge] += 1
                                    # Update this edge
                                    this_edge = next_edge
                                    # Update this density
                                    this_density = self.density(this_edge)
                                    # Update to current agent edge
                                    self.K[this_agent] = this_density
                                    # Calculate new velocity
                                    this_velocity = fd.v_dict[this_density]
                                    # Assign the new distance taking into account the remaining time
                                    this_location = this_velocity * residual_time
                                    # Calculate new progress
                                    di = self.h.D[this_edge]
                            # Calculate the residual distance
                            residual_di = this_location - di
                        # Determine new position
                        this_node = np.array(self.h.nodes[ri])
                        that_node = np.array(self.h.nodes[ci])
                        pr = this_location / di
                        offset = (that_node-this_node) * pr
                        self.P[this_agent,:] = this_node + offset
                        # Update array of unit vectors if the agent marker is quiver
                        if self.agent_marker == 'quiver':
                            magnitude = np.linalg.norm(offset)
                            if magnitude > 0:
                                unit = offset/magnitude
                            else:
                                unit = offset
                            self.U[this_agent,:] = unit
                        # Update to current agent location
                        self.L[this_agent] = this_location
                        # Update to current agent edge
                        self.E[this_agent] = this_edge
                        if this_agent in self.track_list:
                            self.tracked_agent[this_agent].append(self.dist2exit(this_agent))
                        # Update journey time based on velocity on this link
                        if self.use_JT:
                            self.JT[ci][ri]['weight']=di/this_velocity
                        # Get the bin id in which to add the agent
                        bin = fd.which_bin(this_density)
                        # This is the density profile for a given agent over all timestep
                        self.KP_agent[this_agent,bin] = self.KP_agent[this_agent,bin] + 1
                        # This is the density profile for all agents per time step
                        this_tstep[bin] = this_tstep[bin] + 1
                        # This is the density profile for all agents for all edges over all timestep
                        self.KP_edges[this_edge,bin] = self.KP_edges[this_edge,bin] + 1
                    # Append the agglomerate density profile to our list of density profiles per timestep
                    self.KP_tstep.append(this_tstep)
                    # Remove agents that have reached destination
                    for r in removal:
                        self.S.remove(r)
