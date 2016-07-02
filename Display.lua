local classic = require 'classic'
local image = require 'image'

-- Detect QT for image display
local qt = pcall(require, 'qt')

-- Display is responsible for handling QT/recording logic
local Display = classic.class('Display')

-- Creates display; live if using QT
function Display:_init(opt, display)
  self._id = opt._id
  self.zoom = opt.zoom
  self.displayHeight = opt.displaySpec[2][2]
  self.displayWidth = opt.displaySpec[2][3]
  self.saliency = opt.saliency
  self.record = opt.mode == 'eval' and opt.record
  self.fps = 60
  
  -- Activate live display if using QT
  self.window = qt and image.display({image=display, zoom=self.zoom})

  -- Set up recording
  if self.record then
    -- Recreate scratch directory
    paths.rmall('scratch', 'yes')
    paths.mkdir('scratch')

    log.info('Recording screen')
  end

  classic.strict(self)
end

-- Computes saliency map for display from agent field
function Display:createSaliencyMap(agent, display)
  local screen = display:clone() -- Cloned to prevent side-effects
  local saliencyMap = agent.saliencyMap:float()

  -- Use red channel for saliency map
  screen:select(1, 1):copy(image.scale(saliencyMap, self.displayWidth, self.displayHeight))

  return screen
end

-- Show display (handles recording as well for efficiency)
function Display:display(agent, display, step)
  if qt or self.record then
    local screen = self.saliency and self:createSaliencyMap(agent, display) or display

    -- Display
    if qt then
      image.display({image=screen, zoom=self.zoom, win=self.window})
    end

    -- Record
    if self.record then
      image.save(paths.concat('scratch', self._id .. '_' .. string.format('%06d', step) .. '.jpg'), screen)
    end
  end
end

-- Creates videos from frames if recording
function Display:createVideo()
  if self.record then
    log.info('Recorded screen')

    -- Create videos directory
    if not paths.dirp('videos') then
      paths.mkdir('videos')
    end

    -- Use FFmpeg to create a video from the screens
    log.info('Creating video')
    os.execute('ffmpeg -framerate ' .. self.fps .. ' -start_number 1 -i scratch/' .. self._id .. '_%06d.jpg -c:v libvpx-vp9 -crf 0 -b:v 0 videos/' .. self._id .. '.webm')
    log.info('Created video')

    -- Clear scratch space
    paths.rmall('scratch', 'yes')
  end
end

return Display
